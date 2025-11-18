import argparse
import sys

class CLI:
    """CLI describes a command line interface for interacting.
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="cli runner",)
        parser.add_argument("command", help="Subcommand to run")
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):
        import train

        train.cli(sys.argv[2:])

    def predict(self):
        import predict

        predict.cli(sys.argv[2:])

    def evaluate(self):
        import evaluate

        evaluate.cli(sys.argv[2:])

    def gate(self):
        import gate

        gate.cli(sys.argv[2:])

    def serve(self):
        import serve

        serve.cli(sys.argv[2:])

def main():
    CLI()

if __name__ == "__main__":
    main()
