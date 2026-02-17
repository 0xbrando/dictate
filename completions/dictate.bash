# Bash completion for dictate
# Install: source this file or copy to /etc/bash_completion.d/dictate
# Or add to ~/.bashrc: eval "$(dictate --completions bash)" (future)

_dictate_completions() {
    local cur prev words cword
    _init_completion || return

    local commands="config stats status doctor devices update"
    local config_subcmds="show set reset path"
    local config_keys="writing_style quality stt input_language output_language ptt_key command_key llm_cleanup sound llm_endpoint advanced_mode"
    local writing_styles="clean formal bullets email slack technical tweet raw"
    local quality_values="0 1 2 3 4 api speedy fast balanced quality"
    local stt_values="0 1 parakeet whisper"
    local languages="auto en pl de fr es it pt nl ja zh ko ru"
    local ptt_keys="ctrl_l ctrl_r cmd_r alt_l alt_r"
    local command_keys="none alt_r alt_l cmd_r ctrl_r"
    local bool_values="on off"
    local sound_values="0 1 2 3 4 5 6 soft_pop chime warm click marimba simple"

    case "${cword}" in
        1)
            COMPREPLY=( $(compgen -W "${commands} --help --version --foreground -h -V -f" -- "${cur}") )
            return
            ;;
    esac

    case "${words[1]}" in
        config)
            case "${cword}" in
                2)
                    COMPREPLY=( $(compgen -W "${config_subcmds}" -- "${cur}") )
                    return
                    ;;
                3)
                    if [[ "${words[2]}" == "set" ]]; then
                        COMPREPLY=( $(compgen -W "${config_keys}" -- "${cur}") )
                        return
                    fi
                    ;;
                4)
                    if [[ "${words[2]}" == "set" ]]; then
                        case "${words[3]}" in
                            writing_style) COMPREPLY=( $(compgen -W "${writing_styles}" -- "${cur}") ) ;;
                            quality) COMPREPLY=( $(compgen -W "${quality_values}" -- "${cur}") ) ;;
                            stt) COMPREPLY=( $(compgen -W "${stt_values}" -- "${cur}") ) ;;
                            input_language) COMPREPLY=( $(compgen -W "${languages}" -- "${cur}") ) ;;
                            output_language) COMPREPLY=( $(compgen -W "${languages}" -- "${cur}") ) ;;
                            ptt_key) COMPREPLY=( $(compgen -W "${ptt_keys}" -- "${cur}") ) ;;
                            command_key) COMPREPLY=( $(compgen -W "${command_keys}" -- "${cur}") ) ;;
                            llm_cleanup) COMPREPLY=( $(compgen -W "${bool_values}" -- "${cur}") ) ;;
                            sound) COMPREPLY=( $(compgen -W "${sound_values}" -- "${cur}") ) ;;
                            advanced_mode) COMPREPLY=( $(compgen -W "${bool_values}" -- "${cur}") ) ;;
                        esac
                        return
                    fi
                    ;;
            esac
            ;;
    esac
}

complete -F _dictate_completions dictate
