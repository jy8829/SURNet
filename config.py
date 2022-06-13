import argparse

def args_setting():
    # Training settings
    parser = argparse.ArgumentParser(description='Fish segmentation')
    parser.add_argument('-b', '--batch-size', type=int, default=5, metavar='N',
                        help='input batch size for training (default: 5)')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('-n', '--dataset_num', type=str, default=1000, metavar='N',
                        help='number of dataset to train (default: 1000)')  
    parser.add_argument('-p', '--dataset_place', type=str, default='ori', metavar='N',
                        help='place of dataset to train (default: mosaic)') 
    parser.add_argument('-l', '--loss', type=str, default='BCE',
                        help='loss name(defaulf : BCE)')   
    parser.add_argument('-m', '--model', type=str, default='vgg16',
                        help='model name(defaulf : RUSnet)')                                        
    args = parser.parse_args()
    return args
