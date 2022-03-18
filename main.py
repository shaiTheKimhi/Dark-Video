import torch
import os
from model import *
from dataset import *
from train import *
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#d = torch.load('models/reg_best.pt')
#print(d['psnr'])
#print(d["epoch"])


def eval_model(model, dl, name="regular"):
    total_psnr = 0.0
    total_tpsnr = 0.0
    total_samples = 0
    bar = tqdm(dl)
    count = 0
    for x1, x2, y in bar: #can save one random image
        torch.cuda.empty_cache()
        model.eval()
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        total_samples += 1
        y1, y2 = model(x1), model(x2)
        total_psnr += psnr(y1, y).item()
        total_tpsnr += psnr(y1, y2).item()
        
        name = name.split(".pt")[0].split("_")[0]
        bar.set_description(f"model name:{name}, psnr:{str(round(total_psnr / total_samples, 4))}, tpsnr:{str(round(total_tpsnr / total_samples, 4))}", refresh=True)
        #can add SSIM evaluation here

    return (total_psnr / total_samples), (total_tpsnr / total_samples)


caller_count = 0
caller_images = None
def get_images(model, dl, name):
    global caller_count
    global caller_images
    name = name.split(".pt")[0]


    if caller_count == 0:
        x1, _, y = [i for i in dl][6]
        caller_images = (x1, y)

        plt.imshow(x1[0].permute(1,2,0))
        plt.savefig(os.path.join("logs/images","_origin.png"))

        plt.imshow(y[0].permute(1,2,0))
        plt.savefig(os.path.join("logs/images","_gt.png"))

        caller_count = 1
    
    x1, y = caller_images
    x1, y = x1.to(device), y.to(device)

    plt.imshow(model(x1)[0].permute(1,2,0).cpu().detach().numpy())
    plt.savefig(os.path.join("logs/images",name + "_approx.png"))

    


#check all models on validation set for TPSNR and PSNR for best model saved
#TPSNR: PSNR values between two random frames
if __name__ == "__main__":
    vals = []
    files = [f for f in os.listdir("models") if (".pt" in f and "l0.2" not in f)]

    train_ds, test_ds = create_dataset(crop_size=896) #can change crop size here
    test_dl = torch.utils.data.DataLoader(test_ds,batch_size=1, shuffle=False)
    
    files = ["naive_mom.pt", "naive.pt", "momentum_best.pt", "reg_best.pt", "transfer-2.pt"]

    for file in files:
        torch.cuda.empty_cache()
        data = torch.load(os.path.join("models", file), map_location=torch.device(device))
        if "transfer" in file:
            model = Fcn_resent50().to(device)
        else:
            model = ResUnet().to(device)
        model.load_state_dict(data["model_state_dict"])

        #vals.append(eval_model(model, test_dl, name=file))
        
        get_images(model, test_dl, name=file)

        ###vals.append(valid_epoch(vgg, model, test_dl, F_loss, 0, 1, None).clone().detach().to('cpu')) #lambda and loss function are non-important in that case

        #add TPSNR evaluation (possibly SSIM)

    '''
    file = open("logs/evaluation.txt", "w")
    for i in range(len(files)):
        s = f"model:{files[i]}, psnr:{vals[i][0]}, tpsnr:{vals[i][1]}"
        print(s)
        file.write(s)

    file.close()
    '''







