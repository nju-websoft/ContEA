from model import Encoder_Model
import warnings
from eval import Evaluate
import argparse
import time
from utils import *

warnings.filterwarnings('ignore')

from data_loader import KGs
import numpy as np
import logging


def train_base(args, kgs, model:Encoder_Model, evaluator:Evaluate):
    total_train_time = 0.0
    best_pct = 0.0
    last_pct = 0.0
    degrade_count = 0
    for epoch in range(args.epoch):
        time1 = time.time()
        total_loss = 0
        np.random.shuffle(train_pair)
        batch_num = len(train_pair) // args.batch_size + 1
        model.train()
        for b in range(batch_num):
            pairs = train_pair[b * args.batch_size:(b + 1) * args.batch_size]
            if len(pairs) == 0:
                continue
            pairs = torch.from_numpy(pairs).to(device)
            optimizer.zero_grad()
            loss = model(pairs, None)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        time2 = time.time()
        total_train_time += time2 - time1

        logging.info(
            f'[epoch {epoch + 1}/{args.epoch}]  epoch loss: {(total_loss):.5f}, time cost: {(time2 - time1):.3f}s')

        # Validation
        target_candidate_entities_2 = kgs.entity2 - set(train_pair[:, 1]) - set(valid_pair[:, 1])
        if (epoch + 1) % args.dev_interval == 0:
            logging.info("---------Validation---------")
            model.eval()
            with torch.no_grad():
                candidate = np.array(list(target_candidate_entities_2 - set(valid_pair[:, 1]) - set(train_pair[:, 1])))
                candidate = np.concatenate((valid_pair[:, 1], candidate))
                Lvec, Rvec = model.get_embeddings(valid_pair[:, 0], candidate)
                logging.info(f"(KG1 --> KG2) KG1 valid entity size={len(valid_pair[:, 0])}, KG2 candidate entity size={len(candidate)}")
                P, R, F1 = evaluator.valid(Lvec, Rvec, valid_pair[:, 0], candidate)

            # save model
            if F1 > best_pct:
                best_pct = F1
                logging.info(f"(saving current model to {save_path})\n")
                torch.save(model.state_dict(), save_path)
                degrade_count = 0
            elif F1 <= last_pct:
                logging.info('')
                degrade_count += 1
            elif F1 > last_pct:
                logging.info('')
                degrade_count = 0

            last_pct = F1

            if degrade_count >= args.stop_step:
                logging.info("Early stopped!\n")
                break

    logging.info(f"Total training time: {total_train_time:.3f}s")


def finetune_batch(args, kgs, model:Encoder_Model, evaluator:Evaluate, train_pair, load_path):
    logging.info(f"loading model from {load_path}")
    model.load_state_dict(torch.load(load_path, map_location=device), strict=False)

    new_ent_neighs = torch.from_numpy(kgs.new_ent_nei)
    # generate embeddings for new entities
    model.generate_new_features(new_ent_neighs)

    # freeze model parameters except gate and proxy
    for param in model.parameters():
        param.requires_grad = False
    model.r_encoder.proxy.requires_grad_(True)
    model.e_encoder.proxy.requires_grad_(True)
    model.r_encoder.gate.requires_grad_(True)
    model.e_encoder.gate.requires_grad_(True)
    model.trainable_new_ent_embedding.weight.requires_grad_(True)

    model.train()

    model.print_all_model_parameters()

    # start finetuning
    credible_pair = np.array(kgs.credible_pairs)
    finetune_interval = 1
    total_finetune_time = 0.0
    best_pct = 0.0
    last_pct = 0.0
    degrade_count = 0
    partial_train_pair = np.array(kgs.train_pairs_with_new)
    for epoch in range(args.epoch):
        time1 = time.time()
        total_loss = 0
        np.random.shuffle(partial_train_pair)
        np.random.shuffle(credible_pair)
        logging.info(f"partial train pairs={len(partial_train_pair)}, credible_pairs={len(credible_pair)}")
        batch_num = len(partial_train_pair) // args.batch_size + 1
        credible_batch_size = len(credible_pair) // batch_num + 1
        logging.info(f"train pairs:credible pairs={(args.batch_size/credible_batch_size):.2f}:1")
        for b in range(batch_num):
            pairs1 = partial_train_pair[b * args.batch_size:(b + 1) * args.batch_size]
            pairs2 = credible_pair[b * credible_batch_size:(b + 1) * credible_batch_size]
            if len(pairs1) == 0:
                continue
            pairs1 = torch.from_numpy(pairs1).to(device)
            pairs2 = torch.from_numpy(pairs2).to(device)
            optimizer.zero_grad()
            loss = model(pairs1, pairs2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.ent_embedding.weight.data[model.old_node_size:] = model.trainable_new_ent_embedding.weight.data[model.old_node_size:]

        time2 = time.time()
        total_finetune_time += time2 - time1

        logging.info(
            f'[epoch {epoch + 1}/{args.epoch}]  epoch loss: {(total_loss):.5f}, time cost: {(time2 - time1):.3f}s')

        # Validation
        target_candidate_entities_1 = kgs.entity1 - set(train_pair[:, 0]) - set(valid_pair[:, 0])
        target_candidate_entities_2 = kgs.entity2 - set(train_pair[:, 1]) - set(valid_pair[:, 1])
        if (epoch + 1) % finetune_interval == 0:
            logging.info("---------Validation---------")
            model.eval()
            with torch.no_grad():
                candidate = np.array(list(target_candidate_entities_2 - set(valid_pair[:, 1]) - set(train_pair[:, 1])))
                candidate = np.concatenate((valid_pair[:, 1], candidate))
                Lvec, Rvec = model.get_embeddings(valid_pair[:, 0], candidate)
                logging.info(f"(KG1 --> KG2) KG1 valid entity size={len(valid_pair[:, 0])}, KG2 candidate entity size={len(candidate)}")
                P, R, F1 = evaluator.valid(Lvec, Rvec, valid_pair[:, 0], candidate)

            # save model
            if F1 > best_pct:
                best_pct = F1
                logging.info(f"(saving current model to {save_path})\n")
                torch.save(model.state_dict(), save_path)
                degrade_count = 0
            elif F1 <= last_pct:
                logging.info('')
                degrade_count += 1
            elif F1 > last_pct:
                logging.info('')
                degrade_count = 0

            last_pct = F1

            if degrade_count >= args.stop_step:
                logging.info("Early stopped!\n")
                break

    logging.info(f"Total finetuning time: {total_finetune_time:.3f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='alignment model')
    parser.add_argument('--save_path', default='../saved_model', type=str)
    parser.add_argument('--load_path', default='ZH-EN_base_dim100_0505-1537', type=str)
    parser.add_argument('--log_path', default='../logs', type=str)
    parser.add_argument('--dataset', default='ZH-EN', type=str)
    parser.add_argument('--batch', default='batch1', type=str)
    parser.add_argument('--gpu', default=0, type=int)

    # training and finetuning hyper-parameters
    parser.add_argument('--ent_hidden', default=100, type=int)
    parser.add_argument('--rel_hidden', default=100, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--ind_dropout_rate', default=0.3, type=float)

    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--depth', default=2, type=int)
    parser.add_argument('--gamma', default=2.0, type=float)

    # hyper-parameters for test
    parser.add_argument('--eval_batch_size', default=512, type=int)
    parser.add_argument('--dev_interval', default=2, type=int)
    parser.add_argument('--stop_step', default=3, type=int)
    parser.add_argument('--sim_threshold', default=0.0, type=float)
    parser.add_argument('--topk', default=1, type=int)
    parser.add_argument('--M', default=500, type=int)

    args = parser.parse_args()

    # set logging
    set_logging(args)
    # print hyper-parameters
    print_args(args)
    # set gpu
    device = set_device(args)

    save_path = args.save_path + '/' + args.dataset + '_' + args.batch + '_dim' + \
                str(args.ent_hidden) + '_' + datetime.now().strftime("%m%d-%H%M")
    load_path = args.save_path + '/' + args.load_path

    # load data
    kgs = KGs()
    train_pair, valid_pair, test_pair, ent_adj, r_index, r_val, ent_adj_with_loop, ent_rel_adj = kgs.load_data(args)

    ent_adj = torch.from_numpy(np.transpose(ent_adj))
    ent_rel_adj = torch.from_numpy(np.transpose(ent_rel_adj))
    ent_adj_with_loop = torch.from_numpy(np.transpose(ent_adj_with_loop))
    r_index = torch.from_numpy(np.transpose(r_index))
    r_val = torch.from_numpy(r_val)

    # define model
    model = Encoder_Model(node_hidden=args.ent_hidden,
                          rel_hidden=args.rel_hidden,
                          node_size=kgs.old_ent_num,
                          new_node_size=kgs.total_ent_num,
                          rel_size=kgs.total_rel_num,
                          triple_size=kgs.triple_num,
                          device=device,
                          adj_matrix=ent_adj,
                          new_ent_nei=kgs.new_ent_nei,
                          r_index=r_index,
                          r_val=r_val,
                          rel_matrix=ent_rel_adj,
                          ent_matrix=ent_adj_with_loop,
                          dropout_rate=args.dropout_rate,
                          ind_dropout_rate=args.ind_dropout_rate,
                          gamma=args.gamma,
                          lr=args.lr,
                          depth=args.depth,
                          alpha=args.alpha,
                          beta=args.beta).to(device)

    # define evaluator
    evaluator = Evaluate(test_pairs=kgs.test_pairs,
                         new_test_pairs=kgs.new_test_pairs,
                         new_ent=kgs.new_ent,
                         valid_pairs=kgs.valid_pairs,
                         device=device,
                         eval_batch_size=args.eval_batch_size,
                         k=args.topk,
                         dataset=args.dataset,
                         batch=args.batch,
                         M=args.M)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    # print model parameters
    model.print_all_model_parameters()

    if 'base' in args.batch:
        train_base(args, kgs, model, evaluator)
    else:
        finetune_batch(args, kgs, model, evaluator, train_pair, load_path)

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    logging.info(f"\n---------Testing---------")
    with torch.no_grad():
        kg1_test_entities = list(kgs.entity1 - set(train_pair[:, 0]) - set(valid_pair[:, 0]))
        kg2_test_entities = list(kgs.entity2 - set(train_pair[:, 1]) - set(valid_pair[:, 1]))
        logging.info(f"KG1 test entity size={len(kg1_test_entities)}, KG2 test entity size={len(kg2_test_entities)}")

        Lvec, Rvec = model.get_embeddings(kg1_test_entities, kg2_test_entities)

        logging.info(f"KG1 <--> KG2")
        evaluator.test_with_threshold(Lvec, Rvec, kg1_test_entities, kg2_test_entities)
