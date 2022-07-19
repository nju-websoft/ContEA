import numpy as np
import scipy.sparse as sp
import logging
import os


class KGs:
    def __init__(self):
        self.entity1, self.rel1, self.triples1 = None, None, None
        self.entity2, self.rel2, self.triples2 = None, None, None
        self.train_pairs, self.valid_pairs, self.test_pairs = None, None, None
        self.new_ent = set()
        self.anchors = None
        self.new_test_pairs = []
        self.total_ent_num, self.total_rel_num = 0, 0
        self.triple_num = 0
        self.credible_pairs = None
        self.new_ent_nei = np.array([])

    def load_triples(self, file_name):
        triples = []
        entity = set()
        rel = {0}  # self-loop edge
        for line in open(file_name, 'r'):
            head, r, tail = [int(item) for item in line.split()]
            entity.add(head)
            entity.add(tail)
            rel.add(r + 1)  # here all relation add 1
            triples.append((head, r + 1, tail))
        return entity, rel, triples

    def load_alignment_pair(self, file_name):
        alignment_pair = []
        for line in open(file_name, 'r'):
            e1, e2 = line.split()
            alignment_pair.append((int(e1), int(e2)))
        return alignment_pair

    def get_matrix(self, triples, entity, rel):
        ent_size = max(entity) + 1
        rel_size = (max(rel) + 1)
        logging.info(f"Union graph: ent size={ent_size}, rel_size={rel_size}")

        # row-based list of lists sparse matrix
        adj_matrix = sp.lil_matrix((ent_size, ent_size))
        adj_features = sp.lil_matrix((ent_size, ent_size))
        radj = []
        rel_in = np.zeros((ent_size, rel_size))
        rel_out = np.zeros((ent_size, rel_size))

        # add self-loop
        for i in range(max(entity) + 1):
            adj_features[i, i] = 1

        for h, r, t in triples:
            adj_matrix[h, t] = 1
            adj_matrix[t, h] = 1
            adj_features[h, t] = 1
            adj_features[t, h] = 1
            radj.append([h, t, r])
            radj.append([t, h, r + rel_size])
            rel_out[h][r] += 1
            rel_in[t][r] += 1

        count = -1
        s = set()
        d = {}
        r_index, r_val = [], []
        for h, t, r in sorted(radj, key=lambda x: x[0] * 10e10 + x[1] * 10e5):
            if ' '.join([str(h), str(t)]) in s:
                r_index.append([count, r])
                r_val.append(1)
                d[count] += 1
            else:
                count += 1
                d[count] = 1
                s.add(' '.join([str(h), str(t)]))
                r_index.append([count, r])
                r_val.append(1)
        for i in range(len(r_index)):
            r_val[i] /= d[r_index[i][0]]

        rel_features = np.concatenate([rel_in, rel_out], axis=1)
        rel_features = sp.lil_matrix(rel_features)
        return adj_matrix, np.array(r_index), np.array(r_val), adj_features, rel_features

    def load_base(self, path):
        entity1, rel1, triples1 = self.load_triples(path + 'triples_1')
        entity2, rel2, triples2 = self.load_triples(path + 'triples_2')
        train_pairs = self.load_alignment_pair(path + 'train_links')
        valid_pairs = self.load_alignment_pair(path + 'valid_links')
        test_pairs = self.load_alignment_pair(path + 'test_links')

        return [entity1, rel1, triples1, entity2, rel2, triples2, train_pairs, valid_pairs, test_pairs]

    def load_data(self, args):
        base_data = self.load_base('../datasets/'+args.dataset+'/base/')
        self.entity1, self.rel1, self.triples1 = base_data[0:3]
        self.entity2, self.rel2, self.triples2 = base_data[3:6]
        self.train_pairs, self.valid_pairs, self.test_pairs = base_data[6:]
        self.anchors = set([p[0] for p in self.train_pairs]).union(set([p[1] for p in self.train_pairs]))
        self.old_ent_num = len(self.entity1) + len(self.entity2)
        self.old_ent_set = self.entity1.union(self.entity2)
        if os.path.exists('../datasets/'+args.dataset+'/base/' + 'credible_pairs'):
            self.credible_pairs = self.load_alignment_pair('../datasets/'+args.dataset+'/base/' + 'credible_pairs')
        if 'batch' in args.batch:
            folders = []
            batch_num = int(args.batch[-1])
            for i in range(1, batch_num + 1):
                folders.append('../datasets/'+args.dataset+'/batch' + str(i) + '/')
            for i, f in enumerate(folders):
                entity1_tmp, rel1_tmp, triples1_tmp = self.load_triples(f + 'new_triples_1')
                entity2_tmp, rel2_tmp, triples2_tmp = self.load_triples(f + 'new_triples_2')
                self.entity1 = self.entity1.union(entity1_tmp)
                self.entity2 = self.entity2.union(entity2_tmp)
                self.triples1 = self.triples1 + triples1_tmp
                self.triples2 = self.triples2 + triples2_tmp
                if i < batch_num-1:
                    self.old_ent_num = len(self.entity1) + len(self.entity2)
                    self.old_ent_set = self.entity1.union(self.entity2)
                    self.credible_pairs = self.load_alignment_pair(f + 'credible_pairs')
                else:
                    self.new_ent_nei = []
                    self.new_ent_nei_anchor = set()
                    for tr in triples1_tmp:
                        h, r, t = tr
                        if h not in self.old_ent_set and t in self.old_ent_set:
                            self.new_ent_nei.append((h, t))
                            if t in self.anchors:
                                self.new_ent_nei_anchor.add(t)
                        elif h in self.old_ent_set and t not in self.old_ent_set:
                            self.new_ent_nei.append((t, h))
                            if h in self.anchors:
                                self.new_ent_nei_anchor.add(h)
                    for tr in triples2_tmp:
                        h, r, t = tr
                        if h not in self.old_ent_set and t in self.old_ent_set:
                            self.new_ent_nei.append((h, t))
                            if t in self.anchors:
                                self.new_ent_nei_anchor.add(t)
                        elif h in self.old_ent_set and t not in self.old_ent_set:
                            self.new_ent_nei.append((t, h))
                            if h in self.anchors:
                                self.new_ent_nei_anchor.add(h)
                    self.new_ent_nei = np.transpose(np.array(self.new_ent_nei))
                    self.new_ent = self.new_ent_nei[0]

                    self.train_pairs_with_new = []
                    for p in self.train_pairs:
                        for k in self.new_ent_nei_anchor:
                            if k in p:
                                self.train_pairs_with_new.append(p)
                                break

            self.test_pairs = self.load_alignment_pair('../datasets/'+args.dataset+'/' + args.batch + '/' + 'all_test_links')
            self.new_test_pairs = self.load_alignment_pair('../datasets/'+args.dataset+'/' + args.batch + '/' + 'new_test_links')
            self.old_test_pairs = list(set(self.test_pairs) - set(self.new_test_pairs))

        logging.info(f"KG1 entity num={len(self.entity1)}, relation num={len(self.rel1)}, triple num={len(self.triples1)}")
        logging.info(f"KG2 entity num={len(self.entity2)}, relation num={len(self.rel2)}, triple num={len(self.triples2)}")
        logging.info(f"train pairs: {len(self.train_pairs)}, valid pairs: {len(self.valid_pairs)}, test pairs: {len(self.test_pairs)}, new test pairs: {len(self.new_test_pairs)}")
        if 'batch' in args.batch:
            logging.info(f"KG1+KG2 new entity num={len(self.entity1)+len(self.entity2)-self.old_ent_num}")
            logging.info(f"credible pairs: {len(self.credible_pairs)}")

        adj_matrix, r_index, r_val, adj_features, rel_features = self.get_matrix(self.triples1 + self.triples2,
                                                                                 self.entity1.union(self.entity2),
                                                                                 self.rel1.union(self.rel2))

        ent_adj = np.stack(adj_matrix.nonzero(), axis=1)
        ent_adj_with_loop = np.stack(adj_features.nonzero(), axis=1)
        ent_rel_adj = np.stack(rel_features.nonzero(), axis=1)

        self.total_ent_num = adj_features.shape[0]
        self.total_rel_num = rel_features.shape[1]  # including self-loop edge and inv edge
        self.triple_num = ent_adj.shape[0]

        return np.array(self.train_pairs), np.array(self.valid_pairs), np.array(self.test_pairs), \
               ent_adj, r_index, r_val, ent_adj_with_loop, ent_rel_adj
