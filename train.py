import torch
import torch.nn.functional as F

from utils import *
from trainers import *


def evaluate_model(clip_model, logit_scale, loader, dataset):
    embeddings = []
    targets = []

    clip_model.eval()
    with torch.no_grad():
        template = dataset.template
        texts = [template.format(classname.replace("_", " ")) for classname in dataset.classnames]

    acc = 0.0
    tot_samples = 0

    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = images.cuda(), target.cuda()

            # UPDATED: model returns (img_feat, txt_feat, kl_reg) OR (img_feat, hidden_states, txt_feat, kl_reg)
            out = clip_model(images, texts)
            if len(out) == 4:
                image_features, hidden_states, text_features, kl_reg = out
            else:
                image_features, text_features, kl_reg = out

            embeddings.append(image_features)
            targets.append(target)

            cosine_similarity = logit_scale * (image_features @ text_features.t())
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)

    acc /= tot_samples
    return acc


def run_training(args, clip_model, logit_scale, dataset, train_loader, test_loader):
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    clip_model = clip_model.float().cuda()
    clip_model, _ = build_evi_steer(args)

    if args.eval_only:
        load_model(args, clip_model)
        acc_test = evaluate_model(clip_model, logit_scale, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return acc_test

    print_trainable_parameters(clip_model)

    total_epochs = args.train_epochs

    optimizer = torch.optim.AdamW(
        clip_model.parameters(),
        weight_decay=0,
        betas=(0.9, 0.999),
        lr=args.lr,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 400, eta_min=1e-6
    )

    # KL regularization weight
    lambda_kl = args.lambda_kl

    for train_idx in range(total_epochs):
        clip_model.train()
        acc_train, tot_samples, loss_epoch = 0.0, 0, 0.0
        loss_cls_epoch, loss_kl_epoch = 0.0, 0.0

        for i, (images, target) in enumerate(tqdm(train_loader)):
            images, target = images.cuda(), target.cuda()

            template = dataset.template
            texts = [template.format(classname.replace("_", " ")) for classname in dataset.classnames]

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out = clip_model(images, texts)
                if len(out) == 4:
                    image_features, hidden_states, text_features, kl_reg = out
                else:
                    image_features, text_features, kl_reg = out

                cosine_similarity = logit_scale * (image_features @ text_features.t())

                # classification loss
                loss_cls = F.cross_entropy(cosine_similarity, target)

                loss_kl = kl_reg.float()
                # total loss: CE + lambda_kl * KL
                loss = loss_cls + (lambda_kl * loss_kl)

            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            tot_samples += target.shape[0]

            loss_epoch += loss.item() * target.shape[0]
            loss_cls_epoch += loss_cls.item() * target.shape[0]
            loss_kl_epoch += loss_kl.item() * target.shape[0]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

        acc_train /= tot_samples
        loss_epoch /= tot_samples
        loss_cls_epoch /= tot_samples
        loss_kl_epoch /= tot_samples

        current_lr = scheduler.get_last_lr()[0]
        print(
            "Epoch {}, Acc: {:.4f}, Loss: {:.4f} (CE: {:.4f}, KL: {:.6f}, λ_kl={:.1e})".format(
                train_idx + 1,
                acc_train,
                loss_epoch,
                loss_cls_epoch,
                loss_kl_epoch,
                lambda_kl,
            )
        )

    acc_test = evaluate_model(clip_model, logit_scale, test_loader, dataset)

    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))

    if args.save_path is not None:
        save_model(args, clip_model)

    return acc_test