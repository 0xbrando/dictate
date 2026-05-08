import Foundation
import FluidAudio

// MARK: - Helpers

func printJSON(_ dict: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: dict, options: []),
       let str = String(data: data, encoding: .utf8) {
        print(str)
        fflush(stdout)
    }
}

func exitError(_ message: String) -> Never {
    fputs("error: \(message)\n", stderr)
    exit(1)
}

// MARK: - Transcription

func runTranscribe(args: ArraySlice<String>) async {
    var filePath: String?
    var useStdin = false
    var sampleRate = 16000
    var modelVersion: AsrModelVersion = .v3

    var i = args.startIndex
    while i < args.endIndex {
        let arg = args[i]
        switch arg {
        case "--stdin":
            useStdin = true
        case "--sample-rate":
            i += 1
            guard i < args.endIndex, let rate = Int(args[i]) else {
                exitError("--sample-rate requires an integer value")
            }
            sampleRate = rate
        case "--model-version":
            i += 1
            guard i < args.endIndex else {
                exitError("--model-version requires v2 or v3")
            }
            modelVersion = args[i] == "v2" ? .v2 : .v3
        default:
            if arg.hasPrefix("-") {
                exitError("unknown option: \(arg)")
            }
            filePath = arg
        }
        i += 1
    }

    if !useStdin && filePath == nil {
        exitError("provide a WAV file path or use --stdin")
    }

    fputs("Loading ASR models...\n", stderr)
    let startLoad = CFAbsoluteTimeGetCurrent()

    let models: AsrModels
    do {
        models = try await AsrModels.downloadAndLoad(version: modelVersion)
    } catch {
        exitError("failed to download/load models: \(error.localizedDescription)")
    }

    let asrManager = AsrManager(config: .default)
    do {
        try await asrManager.loadModels(models)
    } catch {
        exitError("failed to initialize ASR: \(error.localizedDescription)")
    }

    let loadTime = CFAbsoluteTimeGetCurrent() - startLoad
    fputs("Models loaded in \(String(format: "%.1f", loadTime))s\n", stderr)

    let startTranscribe = CFAbsoluteTimeGetCurrent()

    let result: ASRResult
    var decoderState = TdtDecoderState.make(decoderLayers: await asrManager.decoderLayerCount)
    do {
        if useStdin {
            let data = FileHandle.standardInput.readDataToEndOfFile()
            let sampleCount = data.count / 2
            var samples = [Float](repeating: 0, count: sampleCount)
            data.withUnsafeBytes { rawBuffer in
                let int16Buffer = rawBuffer.bindMemory(to: Int16.self)
                for j in 0..<sampleCount {
                    samples[j] = Float(int16Buffer[j]) / 32768.0
                }
            }

            if sampleRate != 16000 {
                let ratio = Float(16000) / Float(sampleRate)
                let newCount = Int(Float(sampleCount) * ratio)
                var resampled = [Float](repeating: 0, count: newCount)
                for j in 0..<newCount {
                    let srcIdx = min(Int(Float(j) / ratio), sampleCount - 1)
                    resampled[j] = samples[srcIdx]
                }
                samples = resampled
            }

            result = try await asrManager.transcribe(samples, decoderState: &decoderState)
        } else {
            let url = URL(fileURLWithPath: filePath!)
            guard FileManager.default.fileExists(atPath: filePath!) else {
                exitError("file not found: \(filePath!)")
            }
            result = try await asrManager.transcribe(url, decoderState: &decoderState)
        }
    } catch {
        exitError("transcription failed: \(error.localizedDescription)")
    }

    let transcribeTime = CFAbsoluteTimeGetCurrent() - startTranscribe
    let durationMs = Int(transcribeTime * 1000)

    printJSON([
        "text": result.text,
        "duration_ms": durationMs,
    ] as [String: Any])
}

func loadAsrManager(modelVersion: AsrModelVersion) async -> AsrManager {
    fputs("Loading ASR models...\n", stderr)
    let startLoad = CFAbsoluteTimeGetCurrent()

    let models: AsrModels
    do {
        models = try await AsrModels.downloadAndLoad(version: modelVersion)
    } catch {
        exitError("failed to download/load models: \(error.localizedDescription)")
    }

    let asrManager = AsrManager(config: .default)
    do {
        try await asrManager.loadModels(models)
    } catch {
        exitError("failed to initialize ASR: \(error.localizedDescription)")
    }

    let loadTime = CFAbsoluteTimeGetCurrent() - startLoad
    fputs("Models loaded in \(String(format: "%.1f", loadTime))s\n", stderr)
    return asrManager
}

func transcribeFile(_ filePath: String, using asrManager: AsrManager) async -> [String: Any] {
    let startTranscribe = CFAbsoluteTimeGetCurrent()
    var decoderState = TdtDecoderState.make(decoderLayers: await asrManager.decoderLayerCount)

    do {
        let url = URL(fileURLWithPath: filePath)
        guard FileManager.default.fileExists(atPath: filePath) else {
            return ["error": "file not found: \(filePath)"]
        }
        let result = try await asrManager.transcribe(url, decoderState: &decoderState)
        let transcribeTime = CFAbsoluteTimeGetCurrent() - startTranscribe
        return [
            "text": result.text,
            "duration_ms": Int(transcribeTime * 1000),
        ]
    } catch {
        return ["error": "transcription failed: \(error.localizedDescription)"]
    }
}

func runServe(args: ArraySlice<String>) async {
    var modelVersion: AsrModelVersion = .v3

    var i = args.startIndex
    while i < args.endIndex {
        let arg = args[i]
        switch arg {
        case "--model-version":
            i += 1
            guard i < args.endIndex else {
                exitError("--model-version requires v2 or v3")
            }
            modelVersion = args[i] == "v2" ? .v2 : .v3
        default:
            exitError("unknown option: \(arg)")
        }
        i += 1
    }

    let asrManager = await loadAsrManager(modelVersion: modelVersion)
    printJSON(["ready": true])

    while let line = readLine() {
        guard let data = line.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let path = object["path"] as? String else {
            printJSON(["error": "invalid request"])
            continue
        }

        let result = await transcribeFile(path, using: asrManager)
        printJSON(result)
    }
}

// MARK: - Check

func runCheck() {
    let version = ProcessInfo.processInfo.operatingSystemVersion
    let supported = version.majorVersion >= 14

    #if arch(arm64)
    let arch = "arm64"
    #else
    let arch = "x86_64"
    #endif

    printJSON([
        "available": supported,
        "macos_version": "\(version.majorVersion).\(version.minorVersion).\(version.patchVersion)",
        "arch": arch,
    ])

    exit(supported ? 0 : 1)
}

// MARK: - Entry point

@main
struct DictateSTTCLI {
    static func main() async {
        let arguments = CommandLine.arguments
        guard arguments.count >= 2 else {
            fputs("""
            dictate-stt: ANE-accelerated speech-to-text for Dictate

            Usage:
              dictate-stt transcribe <path.wav> [--model-version v2|v3]
              dictate-stt transcribe --stdin --sample-rate <rate>
              dictate-stt serve [--model-version v2|v3]
              dictate-stt check

            Commands:
              transcribe  Transcribe audio to text (outputs JSON to stdout)
              serve       Keep models loaded and accept JSON-lines {"path": "..."} requests
              check       Check if ANE STT is available on this system

            """, stderr)
            exit(1)
        }

        switch arguments[1] {
        case "transcribe":
            await runTranscribe(args: arguments.dropFirst(2)[...])
        case "serve":
            await runServe(args: arguments.dropFirst(2)[...])
        case "check":
            runCheck()
        default:
            exitError("unknown command: \(arguments[1]). Use 'transcribe', 'serve', or 'check'.")
        }
    }
}
