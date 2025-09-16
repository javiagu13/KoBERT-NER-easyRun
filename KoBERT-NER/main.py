import argparse

from trainer import Trainer
from utils import init_logger, load_tokenizer, set_seed, MODEL_CLASSES, MODEL_PATH_MAP
from data_loader import load_and_cache_examples


def main(args):
    init_logger()
    set_seed(args)
    
    tokenizer = load_tokenizer(args)

    train_dataset = None
    dev_dataset = None
    test_dataset = None

    if args.do_train or args.do_eval:
        test_dataset = load_and_cache_examples(args, tokenizer, mode="test")
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, mode="train")

    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test", "eval")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="naver-ner", type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--pred_dir", default="./preds", type=str, help="The prediction file dir")
    parser.add_argument("--train_file", default="train.tsv", type=str, help="Train file")
    # parser.add_argument("--train_file", default="train_spanner_1x.tsv", type=str, help="Train file") ### This one is for spanner 1x dataset
    # parser.add_argument("--train_file", default="train_tokner_1x.tsv", type=str, help="Train file") ### This one is for tokner 1x dataset
    # parser.add_argument("--train_file", default="train_pcliner_2x.tsv", type=str, help="Train file") ### This one is for pcliner 2x dataset
    parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file")
    parser.add_argument("--label_file", default="label.txt", type=str, help="Slot Label file")
    parser.add_argument("--write_pred", action="store_true", help="Write prediction during evaluation")
    parser.add_argument("--model_type", default="kobert", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys())) ### If changing this double check the path in utils.py to make it local model
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Batch size for training.")  
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for evaluation.")  
    parser.add_argument("--max_seq_len", default=512, type=int, help="BERT’s maximum input length.")  
    parser.add_argument("--learning_rate", default=2e-5, type=float)  
    parser.add_argument("--num_train_epochs", default=16, type=float)  
    parser.add_argument("--weight_decay", default=0.01, type=float)  
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)  
    parser.add_argument("--warmup_steps", default=80, type=int)  
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)  
    parser.add_argument("--max_grad_norm", default=1.0, type=float)  
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=125, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=125, help="Save checkpoint every X updates steps.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    args = parser.parse_args()

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    main(args)
