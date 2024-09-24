import numpy as np
import torch
from tqdm import tqdm as tqdm
import torch.nn.functional as F
from utils import device
from operator import itemgetter

def test(params, model, testset, category):
    model.eval()
    (test_queries, test_targets, name) = (testset.test_queries, testset.test_targets, category)
    with torch.no_grad():
        if test_queries:
            # compute all image features
            index_features = torch.empty((0, model.clip_feature_dim)).to(device, non_blocking=True)
            target_features = torch.empty((0, model.clip_feature_dim)).to(device, non_blocking=True)
            token_index_features = torch.empty((0, model.img_tokens, model.clip_img_feature_dim)).to(device, non_blocking=True)
            
            index_names = []
            imgs = []
            logits = []
            for t in tqdm(test_targets):
                imgs += [t['target_img_data']]
                index_names += [t['image_name']]
                if len(imgs) >= params.batch_size or t is test_targets[-1]:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                    imgs = torch.stack(imgs).float()
                    with torch.no_grad():
                        batch_outputs = model.encode_image(imgs)
                        batch_features = batch_outputs.image_embeds
                        batch_token_features = batch_outputs.last_hidden_state
                        batch_target_features = model.encode_features(batch_token_features, None, None)
                        batch_target_features = model.combine_features(batch_features, None,
                                                       batch_target_features[:, :model.img_tokens], None)
                        target_features = torch.vstack((target_features, batch_target_features))
                        index_features = torch.vstack((index_features, batch_features))
                        token_index_features = torch.vstack((token_index_features, batch_token_features))
                        index_names.extend(index_names)
                        imgs = []
                        index_names = []
            
            # compute test query features
            print("Compute CIRR validation predictions")
            
            # Get a mapping from index names to index features
            name_to_feat = dict(zip(index_names, index_features))
            name_to_feat_token = dict(zip(index_names, token_index_features))
            
            
            # Initialize predicted features, target_names, group_members and reference_names
            predicted_features = torch.empty((0, model.clip_feature_dim)).to(device, non_blocking=True)
            target_names = []
            group_members = []
            reference_names = []
            
            visual_query = []
            textual_query = []
            batch_reference_names = []
            batch_target_names = []
            batch_group_members = []
            
            for t in tqdm(test_queries):
                visual_query += [t['visual_query']]
                textual_query += [t['textual_query']]
                batch_reference_names += [t['reference_name']]
                batch_target_names += [t['target_name']]
                batch_group_members += [t['group_members']]
                
                if len(visual_query) >= params.batch_size or t is test_queries[-1]:
                    visual_query = torch.stack(visual_query).float().cuda()
                    batch_group_members = np.array(batch_group_members).T.tolist()
                    text_inputs = model.tokenizer(textual_query, padding=True, return_tensors='pt').to(device)
                    with torch.no_grad():
                        text_features = model.encode_text(text_inputs)
                        reference_image_features = torch.stack(itemgetter(*batch_reference_names)(name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
                        reference_image_token_features = torch.stack(itemgetter(*batch_reference_names)(name_to_feat_token))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
                        
                        batch_fused_features, _ = model.encode_features(reference_image_token_features, text_features, text_inputs['attention_mask'])
                        batch_predicted_features = model.combine_features(reference_image_features,
                                                                     text_features.text_embeds,
                                                                     batch_fused_features[:, :model.img_tokens],
                                                                     batch_fused_features[:, model.img_tokens:],
                                                                     text_mask=text_inputs['attention_mask'])
                    predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
                    target_names.extend(batch_target_names)
                    group_members.extend(batch_group_members)
                    reference_names.extend(batch_reference_names)
                    
                    visual_query = []
                    textual_query = []
                    batch_reference_names = []
                    batch_target_names = []
                    batch_group_members = []
    return target_features, predicted_features, index_names, reference_names, target_names, group_members

def test_cirr_valset(params, model, testset):
    # Generate predictions
    index_features, predicted_features, index_names, reference_names, target_names, group_members = \
        generate_cirr_val_predictions(params, model, testset)
        
    print("Compute CIRR validation metrics")
    
    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))

    # Compute the subset predictions and ground-truth labels
    group_members = np.array(group_members, dtype=object)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50

def generate_cirr_val_predictions(params, model, testset):
    
    model.eval()
    test_queries, test_targets = testset.val_queries, testset.val_targets
    with torch.no_grad():
        if test_queries:
            # compute all image features
            index_features = torch.empty((0, model.clip_feature_dim)).to(device, non_blocking=True)
            target_features = torch.empty((0, model.clip_feature_dim)).to(device, non_blocking=True)
            token_index_features = torch.empty((0, model.img_tokens, model.clip_img_feature_dim)).to(device, non_blocking=True)
            
            index_all_names = []
            
            index_names = []
            imgs = []
            for t in tqdm(test_targets):
                imgs += [t['target_img_data']]
                index_names += [t['image_name']]
                if len(imgs) >= params.batch_size or t is test_targets[-1]:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                    imgs = torch.stack(imgs).float().to(device, non_blocking=True)
                    with torch.no_grad():
                        batch_outputs = model.encode_image(imgs)
                        batch_features = batch_outputs.image_embeds
                        batch_token_features = batch_outputs.last_hidden_state
                        batch_target_features = model.encode_features(batch_token_features, None, None)
                        batch_target_features = model.combine_features(batch_features, None,
                                                       batch_target_features[:, :model.img_tokens], None)
                        target_features = torch.vstack((target_features, batch_target_features))
                        index_features = torch.vstack((index_features, batch_features))
                        token_index_features = torch.vstack((token_index_features, batch_token_features))
                        index_all_names.extend(index_names)
                        imgs = []
                        index_names = []
            
            # compute test query features
            print("Compute CIRR validation predictions")
            
            # Get a mapping from index names to index features
            name_to_feat = dict(zip(index_all_names, index_features))
            name_to_feat_token = dict(zip(index_all_names, token_index_features))
            
            
            # Initialize predicted features, target_names, group_members and reference_names
            predicted_features = torch.empty((0, model.clip_feature_dim)).to(device, non_blocking=True)
            target_names = []
            group_members = []
            reference_names = []
            
            textual_query = []
            batch_reference_names = []
            batch_target_names = []
            batch_group_members = []
            
            for t in tqdm(test_queries):
                textual_query += [t['textual_query']]
                batch_reference_names += [t['reference_name']]
                batch_target_names += [t['target_name']]
                batch_group_members += [t['group_members']]
                
                if len(textual_query) >= params.batch_size or t is test_queries[-1]:
                    # batch_group_members = np.array(batch_group_members).T.tolist()
                    text_inputs = model.tokenizer(textual_query, padding=True, return_tensors='pt').to(device)
                    with torch.no_grad():
                        text_features = model.encode_text(text_inputs)
                        reference_image_features = torch.stack(itemgetter(*batch_reference_names)(name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
                        reference_image_token_features = torch.stack(itemgetter(*batch_reference_names)(name_to_feat_token))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
                        
                        batch_fused_features, _ = model.encode_features(reference_image_token_features, text_features, text_inputs['attention_mask'])
                        batch_predicted_features = model.combine_features(reference_image_features,
                                                                     text_features.text_embeds,
                                                                     batch_fused_features[:, :model.img_tokens],
                                                                     batch_fused_features[:, model.img_tokens:],
                                                                     text_mask=text_inputs['attention_mask'])
                    predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
                    target_names.extend(batch_target_names)
                    group_members.extend(batch_group_members)
                    reference_names.extend(batch_reference_names)
                    
                    textual_query = []
                    batch_reference_names = []
                    batch_target_names = []
                    batch_group_members = []
    return target_features, predicted_features, index_all_names, reference_names, target_names, group_members


def test_fashion200k_dataset(params, model, testset):
    """Tests a model over the given testset."""
    
    model.eval()
    test_queries = testset.get_test_queries()
    with torch.no_grad():
        all_imgs = []
        all_captions = []
        all_queries = []
        all_target_captions = []
        if test_queries:
            # compute test query features
            imgs = []
     
            visual_query = []
            textual_query = []
            for t in tqdm(test_queries):
                visual_query += [testset.get_written_img(t['source_img_id'], t['target_word'])]
                textual_query += [t['source_caption'] + ', but ' + t['mod']['str']]

                if len(visual_query) >= params.batch_size or t is test_queries[-1]:
                    visual_query = torch.stack(visual_query).float().cuda()
                    f = model.extract_query(textual_query, visual_query).data.cpu().numpy()
                    all_queries += [f]
                    imgs = []
                    visual_query = []
                    textual_query = []
            all_queries = np.concatenate(all_queries)
            all_target_captions = [t['target_caption'] for t in test_queries]

            # compute all image features
            imgs = []
            for i in tqdm(range(len(testset.imgs))):
                imgs += [testset.get_img(i)]
                if len(imgs) >= params.batch_size or i == len(testset.imgs) - 1:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                    imgs = torch.stack(imgs).float().cuda()
                    imgs = model.extract_target(imgs).data.cpu().numpy()
                    all_imgs += [imgs]
                    imgs = []
            all_imgs = np.concatenate(all_imgs)
            all_captions = [img['captions'][0] for img in testset.imgs]

        # feature normalization
        for i in range(all_queries.shape[0]):
            all_queries[i, :] /= np.linalg.norm(all_queries[i, :])

        for i in range(all_imgs.shape[0]):
            all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

        # match test queries to target images, get nearest neighbors
        sims = all_queries.dot(all_imgs.T)
        if test_queries:
            for i, t in enumerate(test_queries):
                sims[i, t['source_img_id']] = -10e10  
        nn_result = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]

        # compute recalls
        out = []
        nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
        for k in [1, 10, 50]:
            r = 0.0
            for i, nns in enumerate(nn_result):
                if all_target_captions[i] in nns[:k]:
                    r += 1
            r /= len(nn_result)
            out += [('recall_top' + str(k) + '_correct_composition', r)]

        return out