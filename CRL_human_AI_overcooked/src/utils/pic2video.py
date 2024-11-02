import imageio
import os

def create_gif(image_list, gif_name, duration = 1.0):
    '''
    :param image_list: 这个列表用于存放生成动图的图片
    :param gif_name: 字符串，所生成gif文件名，带.gif后缀
    :param duration: 图像间隔时间
    :return:
    '''
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

def write_gif(gif_name = r"save.gif", pic_path = r"./output",duration = 0.5):
      
    image_list=[]
    pic_name = os.listdir(pic_path)
    try:
        pic_name_dict = {int(n.split('_')[0]): n for n in pic_name}
    except:
        pic_name_dict = {int(n.split('.')[0]): n for n in pic_name}
    order = list(pic_name_dict.keys())
    order.sort()
    pic_name = [pic_name_dict[k] for k in order]

    for name in pic_name:
        path=os.path.join(pic_path, name)
        image_list.append(path)
    create_gif(image_list, gif_name, duration)
    
if __name__ == '__main__':
    
    gif_name = './save.gif'
    pic_path = '/home/ubuntu/zhanglichao/code/pengyao/MARL_Diversity_2/save_replay/harvest_patch/more_agent/masia_diversity/episode_3/'
    duration = 0.3
    write_gif(gif_name, pic_path, duration)