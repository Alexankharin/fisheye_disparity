import cv2
import numpy as np
import torch
from models.hsm import HSMNet
import matplotlib.pyplot as plt
import ffmpeg
import time
import warnings
import argparse

warnings.filterwarnings("ignore", category=DeprecationWarning) 
parser = argparse.ArgumentParser(description='''
Disparity mapper. Inputs may be equirect or Fisheye videos 
Returned video is inversed depth.
Actual depth can be calculated from disparity map as 
D=IPD*contrastfactor/(I),
where IPD is interpapillary distance, I is intensity (from 0 to 1) 
 ''')

parser.add_argument('--inp','--i', required=True, help='input path')
parser.add_argument('--out','--o', required=True, help='output path')
parser.add_argument('--resolution', '--r', type=int, default=384, help='output resolution')
parser.add_argument('--fovx','--Fovx',  default=180, help='FOVX of input video')
parser.add_argument('--fovy','--Fovy',  default=180, help='FOVY of input video')
parser.add_argument('--projectiontype', '--projectiontype',  default='fisheye', help='input format - fisheye or equirect')
parser.add_argument('--contrastfactor', '--contrast',  default=16, help='multiply inverse depth by this value for better contrast')


args = parser.parse_args()
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getgrid(W,H,fovx=180,fovy=180):
    xs = torch.linspace(-1, 1, steps=W).cuda()
    ys = torch.linspace(-1, 1, steps=H).cuda()
    scalex=3.141592*(fovx/360)
    scaley=3.141592*(fovy/360)
    grid_x = scalex*torch.meshgrid(ys, xs)[1]
    grid_y=scaley*torch.meshgrid(ys, xs)[0]
    #XCALC
    return grid_x, grid_y
def adddisptoframe(frame, disp):
    H,W=frame.shape[:2]
    #W=W//2
    xv, yv = np.meshgrid(range(H//4), range(H//4), indexing='ij')
    MASK=((4*xv/H-0.5)**2+(4*yv/H-0.5)**2<=0.25).reshape(H//4,H//4,1)
    #print(MASK.shape)
    #return MASK
    tinydisp=cv2.resize(disp,(H//4,H//4))
    frame[-MASK.shape[0]:,W//2-W//16:W//2-W//16+MASK.shape[1],...]=frame[-MASK.shape[0]:,W//2-W//16:W//2-W//16+MASK.shape[1],...]*(~MASK)+MASK*tinydisp
    return frame
def createprocess2(resolution=256,h_fov=180,v_fov=180):
    process2 = (ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(resolution, resolution))
        .filter('transpose','1')
        .filter('v360','equirect','fisheye',roll=180,ih_fov=180,iv_fov=180,h_fov=h_fov,v_fov=v_fov)
        .filter('transpose','1')
        .filter('scale','iw/1','ih*2')
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True,pipe_stdin=True, quiet=True)
    )
    return process2

def createprocess2equirect(resolution=256,h_fov=180,v_fov=180):
    process2 = (ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(resolution, resolution))
        .filter('transpose','1')
        .filter('v360','equirect','fisheye',roll=180,ih_fov=180,iv_fov=180,h_fov=180,v_fov=180)
        .filter('transpose','1')
        .filter('scale','iw/1','ih*2')
        .filter('v360','fisheye','equirect',ih_fov=180,iv_fov=180,h_fov=h_fov,v_fov=v_fov)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True,pipe_stdin=True, quiet=True)
    )
    return process2

class WrapperModel(torch.nn.Module):
    def __init__(self, model):
        super(WrapperModel, self).__init__()
        self.model=model
    def forward(self,L,R):
        Ans=model(L,R.roll(-10,-1))
        return (Ans[0]-10).view(-1,1,Ans.shape[-2],Ans.shape[-1]).repeat(1,3,1,1)

def convertvideo(videopath, outputname, resolution =256, fovx=180, fovy=180, projectiontype='fisheye',contrastfactor=16):
    #print (resolution)
    model=HSMNet(maxdisp=torch.tensor(resolution//2).to(device),clean=5)
    pretrained_dict = torch.load('kitti.tar')
    newdict={}
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
    for k,v in pretrained_dict['state_dict'].items():
        k=k[7:]
        if k not in ['preconvs.0.cbr_unit.0.weight', 'preconvs.0.cbr_unit.1.weight', 'preconvs.0.cbr_unit.1.bias', 'preconvs.0.cbr_unit.1.running_mean', 'preconvs.0.cbr_unit.1.running_var', 'preconvs.1.cbr_unit.0.weight', 'preconvs.1.cbr_unit.1.weight', 'preconvs.1.cbr_unit.1.bias', 'preconvs.1.cbr_unit.1.running_mean', 'preconvs.1.cbr_unit.1.running_var']:
            newdict[k]=v
    for k in ['disp_reg8.disp', 'disp_reg16.disp', 'disp_reg32.disp', 'disp_reg64.disp']:
        newdict[k]=getattr(model,k[:-5]).disp

    #model.load_state_dict(pretrained_dict['state_dict'])
    model.load_state_dict(newdict,strict=False)
    model.eval()
    model.cuda()
    hsm=WrapperModel(model)
    inpname=videopath
    outname=outputname
    resolution=resolution
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if projectiontype=='fisheye':
        process1 = (ffmpeg
                .input(inpname)
                .filter('scale',resolution*2,resolution)
                .filter('v360','fisheye','equirect',in_stereo='sbs',roll=90,ih_fov=fovx,iv_fov=fovy,h_fov=180,out_stereo='tb')
                .filter('transpose','1')
                .filter('scale','iw/1','ih/2')
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run_async(pipe_stdout=True, quiet=False)
            )
    if projectiontype=='equirect':
        process1 = (ffmpeg
                .input(inpname)
                .filter('scale',resolution*2,resolution)
                .filter('v360','equirect','fisheye',in_stereo='sbs',h_fov=180,v_fov=180,out_stereo='sbs')
                .filter('v360','fisheye','equirect',in_stereo='sbs',roll=90,ih_fov=180,iv_fov=180,h_fov=180,out_stereo='tb')
                .filter('transpose','1')
                .filter('scale','iw/1','ih/2')
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run_async(pipe_stdout=True, quiet=False)
            )
    cap=cv2.VideoCapture(inpname)
    shift=resolution//16
    ctr=0
    with torch.no_grad():
        while True:
            ret, frame=cap.read()
            if ctr==0:
                Hin,Win=frame.shape[:-1]
                out=cv2.VideoWriter(outname,fourcc,60,(resolution,resolution))
                grid_x,grid_y= getgrid(resolution,resolution)
            if not ret:
                break
            in_bytes = process1.stdout.read(resolution * resolution*2 * 3)
            equirect_frameinit = (
                np
                .frombuffer(in_bytes, np.uint8)
                .reshape([resolution, resolution*2, 3])
            )
            ctr=ctr+1
            framei=cv2.resize(equirect_frameinit,(resolution*2,resolution))
            framel,framer=framei[:,:framei.shape[1]//2,:],framei[:,framei.shape[1]//2:,:]
            L,R=torch.tensor(framel).permute(2,0,1).unsqueeze(0).cuda()/255,torch.tensor(framer).permute(2,0,1).unsqueeze(0).cuda()/255
            #print(L.shape,R.shape)
            Ansl=((model(R,L.roll(-shift,-1))[0]-shift)[0]/(grid_x.cos().clip(0.05,1))/resolution*255).clip(0,255).unsqueeze(-1).repeat(1,1,3)
            equirect_displ=(Ansl.cpu().numpy()*contrastfactor).clip(0,255).astype(np.uint8)
            #equirect_dispr=Ansr.cpu().numpy().astype(np.uint8)
            #EQUIRECTDISPTOFISHEYEDISP
            if projectiontype=='fisheye':
                process2=createprocess2(resolution,h_fov=fovx,v_fov=fovy)
            if projectiontype=='equirect':
                process2=createprocess2equirect(resolution,h_fov=fovx,v_fov=fovy)
            process2.stdin.write(
            equirect_displ
            .astype(np.uint8)
            .tobytes()
                )
            process2.stdin.close()
            Ans2=process2.stdout.read(resolution*resolution * 3)
            disp=np.frombuffer(Ans2, np.uint8).reshape([resolution,resolution, 3]).copy()
            disp[:,:disp.shape[1]//16,:]=disp[:,disp.shape[1]//16:disp.shape[1]//16+1,:]
            disp[:,-disp.shape[1]//16:,:]=disp[:,-disp.shape[1]//16:-disp.shape[1]//16+1,:]
            #addeddisp=adddisptoframe(frame[...,:], disp*4).astype(np.uint8)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            #ans=f
            cv2.imshow('depth',cv2.resize(disp,(400,400)))
            cv2.imshow('frame',cv2.resize(frame,(800,400)))
            #process3.stdin.write(addeddisp[...,::-1].tobytes())
            out.write((disp*1).astype(np.uint8))
            #print(ctr)

        process1.terminate()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
   convertvideo(args.inp, args.out, resolution=args.resolution,fovx=args.fovx, 
                fovy=args.fovy,projectiontype=args.projectiontype,contrastfactor=args.contrastfactor)