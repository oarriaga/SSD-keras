from datasets import DataManager

data_manager = DataManager('VOC2007', split='test')
gt = data_manager.load_data()
print(len(gt.keys()))
