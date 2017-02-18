from cntk.io import ReaderConfig, ImageDeserializer
import zipfile

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "ResNet", "Python"))
sys.path.append(os.path.join(abs_path, "..", "..", "CNTKv2Python", "Examples"))
from prepare_test_data import prepare_CIFAR10_data
from TrainResNet_CIFAR10 import train_and_evaluate, create_reader

def train_cifar_resnet_for_eval(model_dir, data_dir):

    if not os.path.isdir(model_dir)
	    os.mkdir(model_dir)
    if not os.path.isdir(data_dir)
	    os.mkdir(data_dir)
    base_path = prepare_CIFAR10_data()

    # change dir to locate data.zip correctly
    os.chdir(base_path)
	
	# unzip test images for eval
	with zipfile.ZipFile(zip_path) as myzip:
        for fn in range(5)
            myzip.extrac('data/train/%05d.png'%(fn), data_dir)
  
    reader_train = create_reader(os.path.join(base_path, 'train_map.txt'), os.path.join(base_path, 'CIFAR-10_mean.xml'), True)
    reader_test  = create_reader(os.path.join(base_path, 'test_map.txt'), os.path.join(base_path, 'CIFAR-10_mean.xml'), False)

    train_and_evaluate(reader_train, reader_test, 'resnet20', epoch_size=512, max_epochs=1, None, model_dir)
    return base_path
	
if __name__=='__main__':
    
	train_cifar_resnet_for_eval()



