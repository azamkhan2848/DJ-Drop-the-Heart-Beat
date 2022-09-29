import csv, torch, os
import numpy as np

def loadCSV(csvf):
    """
    return a dict saving the information of csv
    :param splitFile: csv file name
    :return: {label:[file1, file2 ...]}
    """
    dictLabels = {}
    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            filename = row[0]
            label = row[1]

            # append filename to current label
            if label in dictLabels.keys():
                dictLabels[label].append(filename)
            else:
                dictLabels[label] = [filename]
    return dictLabels


def txt_to_numpy(filename, row):
    file = open(filename)
    lines = file.readlines()
    datamat = np.arange(row, dtype=np.float)
    row_count = 0
    for line in lines:
        line = line.strip().split(' ')
        datamat[row_count] = line[0]
        row_count += 1

    return datamat


class ToTensor(object):
    def __call__(self, sample):
        text = sample['IEGM_seg']
        return {
            'IEGM_seg': torch.from_numpy(text),
            'label': sample['label']
        }


class IEGM_DataSET():
    def __init__(self, root_dir, indice_dir, mode, size, transform=None, num_classes=2):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.names_list = []
        self.transform = transform
        self.num_classes = num_classes

        csvdata_all = loadCSV(os.path.join(self.indice_dir, mode + '_indice.csv'))

        for i, (k, v) in enumerate(csvdata_all.items()):
            self.names_list.append(str(k) + ' ' + str(v[0]))

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        text_path = self.root_dir + self.names_list[idx].split(' ')[0]

        if not os.path.isfile(text_path):
            print(text_path + 'does not exist')
            return None

        IEGM_seg = txt_to_numpy(text_path, self.size).reshape(1, self.size, 1)

        if self.num_classes == 2:
            label = int(self.names_list[idx].split(' ')[1])
        elif self.num_classes == 8:
            label_dict = {
                    'AFb': 0,
                    'AFt': 1,
                    'SR': 2,
                    'SVT': 3,
                    'VFb': 4,
                    'VFt': 5,
                    'VPD': 6,
                    'VT': 7
                }
            
            fname = self.names_list[idx].split(' ')[0]
            label_name = fname.split('-')[1]
            
            label = label_dict[label_name]
        else:
            raise NotImplementedError("Number of class is not implemented.")

        
        sample = {'IEGM_seg': IEGM_seg, 'label': label}

        return sample