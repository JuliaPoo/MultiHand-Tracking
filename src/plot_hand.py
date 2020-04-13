import matplotlib.pyplot as plt
import numpy as np

HAND_GRAPH = {
    0:[1,17],
    1:[2,5],
    2:[3],
    3:[4],
    4:[],
    5:[6,9],
    6:[7],
    7:[8],
    8:[],
    9:[10,13],
    10:[11],
    11:[12],
    12:[],
    13:[14,17],
    14:[15],
    15:[16],
    16:[],
    17:[18],
    18:[19],
    19:[20],
    20:[]
}
flatten = lambda l: [item for sublist in l for item in sublist]
def plot_img(img, kp_list, box_list, save=None, size=10, line = [], is_right = None):
    
    plt.figure(figsize=(size,size*img.shape[0]/img.shape[1]))
    plt.xlim((0,img.shape[1]))
    plt.ylim((img.shape[0],0))
    plt.imshow(img)
    
    mode_3D = False
    if len(kp_list[0][0]) == 3: mode_3D = True
    
    plot_kp = True
    if len(kp_list) == 0: 
        kp_list = [None]*len(box_list)
        plot_kp = False
        
    plot_bb = True
    if len(box_list) == 0:
        box_list = [None]*len(kp_list)
        plot_bb = False
    
    for kp, box, idx in zip(kp_list, box_list, range(len(kp_list))):
        if type(kp) == type(None): continue
        
        if plot_bb:
            
            box_plot = np.append(box, box[0]).reshape(-1,2)
            plt.plot(box_plot[:,0], box_plot[:,1], c="cyan")
            plt.plot([box_plot[3,0], box_plot[2,0]], [box_plot[3,1], box_plot[2,1]], c="red")
            
            if type(is_right) == type(None):
                plt.annotate(str(idx), box[0])
            else:
                leftright = "left"
                if is_right[idx]: leftright = "right"
                plt.annotate(leftright, kp[12][:2], c="cyan")
        
        if not plot_kp: continue
            
        lines_x = []
        lines_y = []
        for i in range(20):
            lines_x += flatten([[kp[i,0], kp[j,0], None] for j in HAND_GRAPH[i]])
            lines_y += flatten([[kp[i,1], kp[j,1], None] for j in HAND_GRAPH[i]])
        plt.plot(lines_x, lines_y, c="white")
        
        if mode_3D:
            s = kp[:,2] - min(kp[:,2])
            s = max(s) - s + 5
            plt.scatter(kp[:,0], kp[:,1], s=s, c="white", alpha=.6)
        else:
            plt.scatter(kp[:,0], kp[:,1], s=5, c='white', alpha=.6)
        
    plt.axis("off")
    
    if save:
        plt.savefig(save, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
        
    plt.close()