import torch
import torch.nn.functional as F


def loss_on_every_positive_source(model, queries, sources, pos_indexes, neg_indexes):
    queries_round_1 = []
    pos_indexes_1 = []
    neg_indexes_1 = []
    queries_round_2 = []
    pos_indexes_2 = []
    neg_indexes_2 = []
    queries_round_3 = []
    pos_indexes_3 = []
    neg_indexes_3 = []


    for query_id, pos_sources in enumerate(pos_indexes):
        if len(pos_sources) == 1:
            queries_round_1.append(query_id)
            pos_indexes_1.append(pos_sources[0])
            neg_indexes_1.extend(neg_indexes[query_id])
        elif len(pos_sources) == 2:
            queries_round_1.append(query_id)
            pos_indexes_1.append(pos_sources[0])
            neg_indexes_1.extend(neg_indexes[query_id])
            queries_round_2.append(query_id)
            pos_indexes_2.append(pos_sources[1])
            neg_indexes_2.extend(neg_indexes[query_id])
        elif len(pos_sources) == 3:
            queries_round_1.append(query_id)
            pos_indexes_1.append(pos_sources[0])
            neg_indexes_1.extend(neg_indexes[query_id])
            queries_round_2.append(query_id)
            pos_indexes_2.append(pos_sources[1])
            neg_indexes_2.extend(neg_indexes[query_id])
            queries_round_3.append(query_id)
            pos_indexes_3.append(pos_sources[2])
            neg_indexes_3.extend(neg_indexes[query_id])
        else:
            print("many positive sources", query_id, pos_sources)


    _, embed_size = queries.size()
    bsz_1 = len(queries_round_1)
    bsz_2 = len(queries_round_2)
    bsz_3 = len(queries_round_3)

    embedding_query_1 = torch.index_select(queries, 0, torch.LongTensor(queries_round_1)).view(bsz_1, 1, embed_size)
    embedding_pos_1 = torch.index_select(sources, 0, torch.LongTensor(pos_indexes_1)).view(bsz_1, 1, embed_size)
    embedding_neg_1 = torch.index_select(sources, 0, torch.LongTensor(neg_indexes_1)).view(bsz_1, -1, embed_size)
    list_embeddings_queries = [embedding_query_1]
    list_embeddings_pos = [embedding_pos_1]
    list_embeddings_neg = [embedding_neg_1]

    if bsz_2:
        embedding_query_2 = torch.index_select(queries, 0, torch.LongTensor(queries_round_2)).view(bsz_2, 1, embed_size)
        embedding_pos_2 = torch.index_select(sources, 0, torch.LongTensor(pos_indexes_2)).view(bsz_2, 1, embed_size)
        embedding_neg_2 = torch.index_select(sources, 0, torch.LongTensor(neg_indexes_2)).view(bsz_2, -1, embed_size)
        list_embeddings_queries.append(embedding_query_2)
        list_embeddings_pos.append(embedding_pos_2)
        list_embeddings_neg.append(embedding_neg_2)
    if bsz_3:
        embedding_query_3 = torch.index_select(queries, 0, torch.LongTensor(queries_round_3)).view(bsz_3, 1, embed_size)
        embedding_pos_3 = torch.index_select(sources, 0, torch.LongTensor(pos_indexes_3)).view(bsz_3, 1, embed_size)
        embedding_neg_3 = torch.index_select(sources, 0, torch.LongTensor(neg_indexes_3)).view(bsz_3, -1, embed_size)
        list_embeddings_queries.append(embedding_query_3)
        list_embeddings_pos.append(embedding_pos_3)
        list_embeddings_neg.append(embedding_neg_3)

    # embedding_pos_1 =  torch.index_select(sources, 0, torch.LongTensor(pos_indexes_1)).view(bsz_1, 1, embed_size)
    # embedding_pos_2 = torch.index_select(sources, 0, torch.LongTensor(pos_indexes_2)).view(bsz_2, 1, embed_size)
    # embedding_pos_3 = torch.index_select(sources, 0, torch.LongTensor(pos_indexes_3)).view(bsz_3, 1, embed_size)
    # embedding_neg_1 = torch.index_select(sources, 0, torch.LongTensor(neg_indexes_1)).view(bsz_1, -1, embed_size)
    # embedding_neg_2 = torch.index_select(sources, 0, torch.LongTensor(neg_indexes_2)).view(bsz_2, -1, embed_size)
    # embedding_neg_3 = torch.index_select(sources, 0, torch.LongTensor(neg_indexes_3)).view(bsz_3, -1, embed_size)

    embedding_queries = torch.cat(list_embeddings_queries, dim=0)
    embedding_pos = torch.cat(list_embeddings_pos, dim=0)
    embedding_neg = torch.cat(list_embeddings_neg, dim=0)

    logit_scale = model.logit_scale.exp()

    pos_scores = (embedding_queries * embedding_pos).sum(-1) * logit_scale
    neg_scores = (embedding_queries * embedding_neg).sum(-1) * logit_scale
    logit_matrix = torch.cat([pos_scores, neg_scores], 1)

    lsm = F.log_softmax(logit_matrix, dim=1)
    loss = torch.mean(-1.0 * lsm[:, 0])

    return logit_matrix, loss


def loss_on_all_positive_sources(model, queries, sources, pos_indexes, neg_indexes):
    bsz, emb_dim = queries.size()

    indexes_q_1 = []
    indexes_q_2 = []
    indexes_q_3 = []

    indexes_p_1 = []
    indexes_p_2 = []
    indexes_p_3 = []

    indexes_n_1 = []
    indexes_n_2 = []
    indexes_n_3 = []

    for i, pos_sources in enumerate(pos_indexes):
        if len(pos_sources) == 1:
            indexes_q_1.append(i)
            indexes_p_1.extend(pos_sources)
            indexes_n_1.extend(neg_indexes[i])
        elif len(pos_sources) == 2:
            indexes_q_2.append(i)
            indexes_p_2.extend(pos_sources)
            indexes_n_2.extend(neg_indexes[i])
        elif len(pos_sources) == 3:
            indexes_q_3.append(i)
            indexes_p_3.extend(pos_sources)
            indexes_n_3.extend(neg_indexes[i])
        else:
            print("many positive sources", pos_sources)

    bsz_1, bsz_2, bsz_3 = len(indexes_q_1), len(indexes_q_2), len(indexes_q_3)
    bsz = bsz_1 + bsz_2 + bsz_3
    indexes_q = indexes_q_1 + indexes_q_2 + indexes_q_3
    indexes_n = indexes_n_1 + indexes_n_2 + indexes_n_3


    neg_embeddings = torch.index_select(sources, 0, torch.LongTensor(indexes_n)).view(bsz, -1, emb_dim)  # .cuda()
    query_embeddings = torch.index_select(queries, 0, torch.LongTensor(indexes_q)).view(bsz, 1, emb_dim)
    logit_scale = model.logit_scale.exp()
    neg_scores = (query_embeddings * neg_embeddings).sum(-1) * logit_scale

    pos_embeddings_1 = torch.index_select(sources, 0, torch.LongTensor(indexes_p_1)).view(bsz_1, 1, emb_dim)#.cuda()
    query_embeddings_1 = torch.index_select(queries, 0, torch.LongTensor(indexes_q_1)).view(bsz_1, 1, emb_dim)
    pos_scores_1 = (query_embeddings_1 * pos_embeddings_1).sum(-1) * logit_scale

    pos_embeddings_2 = torch.index_select(sources, 0, torch.LongTensor(indexes_p_2)).view(bsz_2, 2, emb_dim)#.cuda()
    query_embeddings_2 = torch.index_select(queries, 0, torch.LongTensor(indexes_q_2)).view(bsz_2, 1, emb_dim)
    pos_scores_2 = (query_embeddings_2 * pos_embeddings_2).sum(-1) * logit_scale
    pos_scores_2 =  torch.sum(pos_scores_2, dim=1, keepdim=True)

    pos_embeddings_3 = torch.index_select(sources, 0, torch.LongTensor(indexes_p_3)).view(bsz_3, 3, emb_dim)#.cuda()
    query_embeddings_3 = torch.index_select(queries, 0, torch.LongTensor(indexes_q_3)).view(bsz_3, 1, emb_dim)
    pos_scores_3 = (query_embeddings_3 * pos_embeddings_3).sum(-1) * logit_scale
    pos_scores_3 = torch.sum(pos_scores_3, dim=1, keepdim=True)

    pos_scores = torch.cat([pos_scores_1, pos_scores_2, pos_scores_3], dim=0)

    logit_matrix = torch.cat([pos_scores, neg_scores], 1)
    lsm = F.log_softmax(logit_matrix, dim=1)
    loss = torch.mean(-1.0 * lsm[:, 0])
    return logit_matrix, loss



def loss_multihop(model, queries, sources, pos_indexes, neg_indexes):
    query_two = []
    target_pos = []
    query_pos = []
    index_neg = []

    for i, pos_sources in enumerate(pos_indexes):
        if len(pos_sources) == 2:
            query_two.append(i)
            query_pos.extend(pos_sources)
            target_pos.extend([pos_sources[1], pos_sources[0]])
            index_neg.extend(neg_indexes[i])
    _, embed_size = queries.size()
    bsz = len(query_two)
    query_embeddings = torch.index_select(queries, 0, torch.LongTensor(query_two))#.cuda()
    query_embeddings = query_embeddings.repeat_interleave(2, dim=0)
    query_pos_embeddings = torch.index_select(sources, 0, torch.LongTensor(query_pos))#.cuda()
    query_enhanced_embeddings = query_embeddings + query_pos_embeddings

    query_enhanced_embeddings = query_enhanced_embeddings.view(2*bsz, 1, embed_size)

    pos_source_embeddings = torch.index_select(sources, 0, torch.LongTensor(target_pos)).view(2*bsz, 1, embed_size)#.cuda()

    neg_source_embeddings = torch.index_select(sources, 0, torch.LongTensor(index_neg))
    neg_source_embeddings = neg_source_embeddings.view(bsz, -1, embed_size)
    neg_source_embeddings = neg_source_embeddings.repeat_interleave(2, dim=0)
    neg_source_embeddings = neg_source_embeddings.view(2 * bsz, -1, embed_size)

    logit_scale = model.logit_scale.exp()
    pos_scores = (query_enhanced_embeddings * pos_source_embeddings).sum(-1) * logit_scale
    neg_scores = (query_enhanced_embeddings * neg_source_embeddings).sum(-1) * logit_scale
    logit_matrix = torch.cat([pos_scores, neg_scores], 1)
    lsm = F.log_softmax(logit_matrix, dim=1)
    loss = torch.mean(-1.0 * lsm[:, 0])

    return logit_matrix, loss





