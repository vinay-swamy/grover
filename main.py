import random
import pickle
import numpy as np
import torch
from rdkit import RDLogger

from grover.util.parsing import parse_args, get_newest_train_args
from grover.util.utils import create_logger
from task.cross_validate import cross_validate
from task.fingerprint import generate_fingerprints
from task.predict import make_predictions, write_prediction
from task.pretrain import pretrain_model
from grover.data.torchvocab import MolVocab


def setup(seed):
    # frozen random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # setup random seed
    setup(seed=42)
    # Avoid the pylint warning.
    a = MolVocab
    # supress rdkit logger
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    # Initialize MolVocab
    mol_vocab = MolVocab

    args = parse_args()
    if args.parser_name == 'finetune':
        logger = create_logger(name='train', save_dir=args.save_dir, quiet=False)
        cross_validate(args, logger)
    elif args.parser_name == 'pretrain':
        logger = create_logger(name='pretrain', save_dir=args.save_dir)
        pretrain_model(args, logger)
    elif args.parser_name == "eval":
        logger = create_logger(name='eval', save_dir=args.save_dir, quiet=False)
        cross_validate(args, logger)
    elif args.parser_name == 'fingerprint':
        train_args = get_newest_train_args()
        logger = create_logger(name='fingerprint', save_dir=None, quiet=False)
        feas, smiles_df = generate_fingerprints(args, logger)
        feas = torch.tensor(feas)
        print((feas.shape[0] , smiles_df.shape[0]))
        assert feas.shape[0] == smiles_df.shape[0]
       
        for i in range(feas.shape[0]):
            outfile = f"/data/vss2134/pgx_pdc/data/drug_embeddings/grover_node/{smiles_df['cpd_id'].iloc[i]}.pickle"
            with open(outfile, 'wb+') as ofl:
                pickle.dump(feas[i], ofl )
                
    elif args.parser_name == 'predict':
        train_args = get_newest_train_args()
        avg_preds, test_smiles = make_predictions(args, train_args)
        write_prediction(avg_preds, test_smiles, args)
