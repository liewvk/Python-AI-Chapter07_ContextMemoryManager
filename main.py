# main.py

import sys

from ollama import ResponseError

from settings import (
    MODEL_NAME,
    MAX_RECENT_EXCHANGES,
    MAX_USER_CHARS,
    SESSION_FILE
)

from memory_manager import ConversationMemory
from ollama_service import OllamaService


def show_help():
    """Display available commands."""

    print("\nCommands:")
    print("  /help     Show commands")
    print("  /history  Show recent exchanges")
    print("  /summary  Show condensed older context")
    print("  /status   Show current memory statistics")
    print("  /save     Save session as JSON")
    print("  /load     Restore the saved session")
    print("  /clear    Clear memory in this run")
    print("  /delete   Delete saved JSON file")
    print("  /exit     Close the program")


def show_history(memory):
    """Display recent conversation messages."""

    if not memory.recent:
        print("No recent exchanges.")
        return

    print("\nRecent conversation:")

    for item in memory.recent:

        print(
            f"\n{item['role'].title()}: "
            f"{item['content']}"
        )


def show_status(memory, last_response):
    """Display memory and token statistics."""

    print(f"\nModel: {MODEL_NAME}")

    print(
        "Recent exchanges:",
        len(memory.recent) // 2
    )

    print(
        "Summarized exchanges:",
        memory.archived_exchanges
    )

    print(
        "Summary length:",
        len(memory.summary),
        "characters"
    )

    if last_response is not None:

        print(
            "Last prompt tokens:",
            last_response.prompt_eval_count
        )

        print(
            "Last output tokens:",
            last_response.eval_count
        )


def process_command(
    command,
    memory,
    service,
    last_response
):
    """Process a conversation management command."""

    if command == "/help":

        show_help()

    elif command == "/history":

        show_history(memory)

    elif command == "/summary":

        print("\nOlder context summary:")

        print(
            memory.summary or
            "(No summary yet.)"
        )

    elif command == "/status":

        show_status(
            memory,
            last_response
        )

    elif command == "/save":

        print(
            "Saved:",
            memory.save()
        )

    elif command == "/load":

        confirm = input(
            "Replace current memory? Type YES: "
        ).strip()

        if confirm == "YES":

            print(
                "Loaded:",
                memory.load()
            )

        else:

            print("Load canceled.")

    elif command == "/clear":

        confirm = input(
            "Clear active conversation? Type YES: "
        ).strip()

        if confirm == "YES":

            memory.clear()

            print(
                "Current memory cleared; "
                "saved file unchanged."
            )

        else:

            print("Clear canceled.")

    elif command == "/delete":

        confirm = input(
            "Delete saved JSON file? Type YES: "
        ).strip()

        if confirm == "YES":

            print(
                "Deleted:",
                memory.delete_saved_file()
            )

            print(
                "Current in-memory conversation "
                "remains unchanged."
            )

        else:

            print("Deletion canceled.")

    else:

        print(
            "Unknown command. Type /help."
        )


def main():

    # Create application services.
    memory = ConversationMemory()

    service = OllamaService()

    last_response = None

    print("==========================================")
    print("   PYTHON CONVERSATION CONTEXT MANAGER")
    print("   Visual Studio 2026 | Ollama + Qwen 3.5")
    print("==========================================")

    print("Model:", MODEL_NAME)

    print(
        "Recent history limit:",
        MAX_RECENT_EXCHANGES,
        "exchanges"
    )

    print(
        "Python interpreter:",
        sys.executable
    )

    show_help()

    try:

        while True:

            try:

                question = input(
                    "\nYou: "
                ).strip()

            except (
                EOFError,
                KeyboardInterrupt
            ):

                print("\nGoodbye!")
                break

            if not question:
                continue

            if question.lower() == "/exit":

                print("Goodbye!")
                break

            # Process application commands.
            if question.startswith("/"):

                try:

                    process_command(
                        question.lower(),
                        memory,
                        service,
                        last_response
                    )

                except (
                    OSError,
                    ValueError
                ) as exc:

                    print(
                        "Session operation failed:",
                        exc
                    )

                continue

            if len(question) > MAX_USER_CHARS:

                print(
                    f"Question cannot exceed "
                    f"{MAX_USER_CHARS} characters."
                )

                continue

            # Process an ordinary AI question.
            try:

                answer, response = service.answer(
                    memory.summary,
                    memory.recent,
                    question
                )

                print(
                    "\nAssistant:",
                    answer
                )

                last_response = response

                # Record the successful exchange.
                memory.add_exchange(
                    question,
                    answer
                )

                # Compress older history when necessary.
                try:

                    count = memory.compress(
                        service
                    )

                    if count:

                        print(
                            f"\n[Summarized {count} "
                            f"older exchange(s).]"
                        )

                except Exception as exc:

                    print(
                        "\n[Summarization failed; "
                        "older messages were kept.]"
                    )

                    print(
                        "Details:",
                        exc
                    )

                    print(
                        "You can still use /save "
                        "and try another question."
                    )

            except ResponseError as exc:

                print(
                    "Ollama request failed:",
                    exc
                )

            except (
                ConnectionError,
                TimeoutError
            ) as exc:

                print(
                    "Ollama unavailable or timed out:",
                    exc
                )

            except ValueError as exc:

                print(
                    "AI response error:",
                    exc
                )

            except Exception as exc:

                print(
                    "Request failed:",
                    exc
                )

    finally:

        service.close()


if __name__ == "__main__":
    main()
