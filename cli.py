import click

from qa_squad.data.utils import bidaf_create_dataset
from qa_squad.training.dist_bert_trainer import distil_bert_qa_train
from qa_squad.training.trainer import qa_train


@click.group()
def main():
    """
        \b
        \b     _______.  ______      __    __       ___       _______
        \b   /       | /  __  \    |  |  |  |     /   \     |       \\
        \b  |   (----`|  |  |  |   |  |  |  |    /  ^  \    |  .--.  |
        \b   \   \    |  |  |  |   |  |  |  |   /  /_\  \   |  |  |  |
        \b.----)   |   |  `--'  '--.|  `--'  |  /  _____  \  |  '--'  |
        \b|_______/     \_____\_____\\______/  /__/     \__\ |_______/

        \b
        A cli interface for playing with the SQUAD dataset.
        Important environment variables: HF_TOKEN

    """
    pass


@click.group()
def bidaf():
    pass


@click.group()
def distilbert():
    pass


if __name__ == '__main__':
    bidaf.add_command(qa_train)
    bidaf.add_command(bidaf_create_dataset)

    distilbert.add_command(distil_bert_qa_train)

    main.add_command(bidaf)
    main.add_command(distilbert)
    main()