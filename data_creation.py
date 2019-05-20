import glob, cv2
path_gt  = './gtFine/'
path_img = './leftImg8bit/'


## path where you want to store your dataset
path = os.getcwd()
path_gt_new = path+'/gt/'
path_img_new = path+'/images/'

folders = ['train', 'test', 'val']
save_folder = ['training', 'testing', 'validation']

for f in folders:
    print("in "+str(f)+" ...")
    
    folder_gt = sorted(glob.glob(path_gt+f+'/*'))
   
    folder_img = sorted(glob.glob(path_img+f+'/*'))
    
    for i in (folder_gt):
        
        gt_list  = glob.glob(i+'/*labellevel3Ids.png')
        for gt in gt_list:
            img = cv2.imread(gt)
            img[img>0]=255
            img = cv2.bitwise_not(img)
            string = gt.split('/'+f)[1].split('/')[2].split('_')[0]
            cv2.imwrite(path_gt_new+save_folder[tt]+'/'+string+'.png',img)
    
    
    for i in (folder_img):
        img_list = glob.glob(i+'/*.png')
        for img in img_list:
            image = cv2.imread(img)
            string = img.split(f+'/')[1].split('/')[1].split('_')[0]
            cv2.imwrite(path_img_new+save_folder[tt]+'/'+string+'.png',image)
        
   
    
