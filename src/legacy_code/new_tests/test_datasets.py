from datasets import DataManager

classes = ['background', 'dog']
data_manager = DataManager('VOC2007', 'test', classes, False)
data = data_manager.load_data()
print(len(data.keys()))
