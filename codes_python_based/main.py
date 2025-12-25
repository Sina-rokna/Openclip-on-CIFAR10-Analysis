import argparse
import open_clip
import torch 
from clip_model import load_clip
from cifar10 import get_cifar10_loaders
from methods.zero_shot import run_zero_shot
from embeddings.image_embedding import compute_image_embedding
from methods.embed_img_inference import run_linear_probe, run_mlp_probe
from methods.embed_txt_img_inference import run_img_txt_mlp            

def get_args():         
    parser = argparse.ArgumentParser(
        description="Hyperparameters for CLIP with CIFAR-10 dataset"
    )
    parser.add_argument('--freeze_clip', default= True, help= 'whether you want freeze CLIP model during learning heards or not?')
    parser.add_argument('--method', type=str, default='linear_probe_img',
                        choices=['zero_shot', 'linear_probe_img', 'mlp_probe_img', 'img_text_mlp_probe'],
                        help='Which mode?')
    # Our Model CLIP
    parser.add_argument('--clip_model', type=str, default='ViT-B-32', help='specifying desired CLIP model architecture')
    parser.add_argument('--clip_pretrained', type=str, default='laion2b_s34b_b79k', help='CLIP pretrained weights')

    # Dataset / DataLoader
    parser.add_argument('--train_size', type=int, default=45000, help = 'number of samples for train set on CIFAR-10')
    parser.add_argument('--val_size', type=int, default=5000, help = 'number of samples for validation set on CIFAR-10')
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_root', type=str, default='./data')

    parser.add_argument('--template_mode', type=str, default='per_class',
                            choices=['shared', 'per_class'],
                            help='Template strategy: "shared" (all classes use same templates) or '
                                '"per_class" (each class has its own specialized templates)')
    # template strategy
    parser.add_argument('--templates', type=str, nargs='+', default=["a photo of a {classname}.",
                                                                    #  "photo featuring a {classname}.",
                                                                    #  "a typical {classname}.",
                                                                    #  "a natural photo of a {classname}.",
                                                                    #  "a {classname} on display.",
                                                                     ], 
                                                                     help='Shared text templates (used when --template_mode shared)')
                                                                     
    # this template argument is for exploring more specialized templates for each class(as the number of classes is low(10), 
    # we investigate a more specialized template system for each class to increase the accuracy of model if possible!!)            
    parser.add_argument('--per_class_templates', type=str, nargs='+', 
                            default=[
                            "an airplane is flying.",
                            "a car driving on the road.",
                            "a photo of a bird.",
                            "a photo of a cat.",
                            "a photo of a deer.",
                            "a photo of a dog.",
                            "a photo of a frog.",
                            "a horse in a field.",
                            "a ship sailing on the sea.",
                            "a truck on the highway."
                         ], 
                         
                        help='10 specialized templates, one per CIFAR-10 class in order: '
                                'airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. '
                                'Required when --template_mode per_class')
    

            
    # Logistic Regression probe
    parser.add_argument('--logreg_max_iter', type=int, default=1000)
    parser.add_argument('--logreg_C', type=float, default=1.0)

    # MLP probe (image features)
    parser.add_argument('--mlp_hidden_dim', type=int, default=512)
    parser.add_argument('--mlp_lr', type=float, default=1e-3)
    parser.add_argument('--mlp_batch_size', type=int, default=64)
    parser.add_argument('--mlp_epochs', type=int, default=120)

    # Image–Text projection heads
    parser.add_argument('--proj_hidden_dim', type=int, default=512)
    parser.add_argument('--proj_output_dim', type=int, default=512)
    parser.add_argument('--proj_lr', type=float, default=1e-3)
    parser.add_argument('--proj_epochs', type=int, default=7) # it's for when we are in Image-Text Mode. 

    args = parser.parse_args()
    return args

def main(): 
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, preprocess = load_clip(args, device)
    train_loader, val_loader, test_loader, classnames = get_cifar10_loaders(preprocess, args)

    if args.method == 'zero_shot':
        run_zero_shot(model, test_loader, classnames, tokenizer, args, device)

    elif args.method == 'linear_probe_img':
        train_feats, train_labels = compute_image_embedding(model, train_loader, device)
        test_feats, test_labels = compute_image_embedding(model, test_loader, device)
        run_linear_probe(train_feats, train_labels, test_feats, test_labels, args)

    elif args.method == 'mlp_probe_img':
        train_feats, train_labels = compute_image_embedding(model, train_loader, device)
        val_feats, val_labels = compute_image_embedding(model, val_loader, device)
        test_feats, test_labels = compute_image_embedding(model, test_loader, device)

        run_mlp_probe(
        train_feats, train_labels,
        val_feats, val_labels,
        test_feats, test_labels,
        args, device)

    elif args.method == 'img_text_mlp_probe':
        run_img_txt_mlp(
            model,
            tokenizer,
            train_loader,
            val_loader,
            test_loader,
            classnames,
            args,
            device
        )

if __name__ == '__main__':
    main()
