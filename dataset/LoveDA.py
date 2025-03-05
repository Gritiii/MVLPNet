from .base_data1 import Few_Data, Base_Data


class LoveDA_few_dataset(Few_Data):

    class_id = {
                0: 'unlabeled',
                1: 'building',
                2: 'road',
                3: 'water',
                4: 'barren',
                5: 'forest',
                6: 'agriculture'
                }
    
    
    all_class = list(range(1, 7))
    val_class = [list(range(1, 3)), list(range(3, 5)), list(range(5, 7))]

    data_root = '../data/LoveDA'
    train_list ='./lists/LoveDA/train.txt'
    val_list ='./lists/LoveDA/val.txt'

    def __init__(self, split=0, shot=1, dataset='LoveDA', mode='train', ann_type='mask', transform_dict=None,transform_tri=None):
        super().__init__(split, shot, dataset, mode, ann_type, transform_dict, transform_tri)



class LoveDA_base_dataset(Base_Data):
    class_id = {
                0: 'unlabeled',
                1: 'building',
                2: 'road',
                3: 'water',
                4: 'barren',
                5: 'forest',
                6: 'agriculture'
                }
    
 
    all_class = list(range(1, 7))
    val_class = [list(range(1, 3)), list(range(3, 5)), list(range(5, 7))]

    data_root = '../data/LoveDA'
    train_list ='./lists/LoveDA/train.txt'
    val_list ='./lists/LoveDA/val.txt'

    def __init__(self, split=0, shot=1, data_root=None, dataset='LoveDA', mode='train', transform_dict=None):
        super().__init__(split,  data_root, dataset, mode, transform_dict)