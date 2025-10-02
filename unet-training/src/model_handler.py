import os
import torch
import mlflow
import warnings
import numpy as np
import utils.globals as globals
from utils.saving import save_model, save_results, save_image_color_legend
from utils.model_architecture import SegmentationModel
from utils.test_helpers import segmentation_scores
from utils.logging import log_results


eps=1e-7


class ModelHandler():
    def __init__(self):
        config = globals.config

        
        self.model = SegmentationModel()

        # loss
        self.loss_mode = config['model']['loss']

        #GPU
        self.model.cuda()
        if torch.cuda.is_available():
            print('Running on GPU',flush=True)
            self.device = torch.device('cuda')
        else:
            warnings.warn("Running on CPU because no GPU was found!")
            self.device = torch.device('cpu')

    def train(self, trainloader, validateloader,testloader,writer):
        config = globals.config
        model = self.model
        device = self.device
        max_score = 100
        c_weights = config['data']['class_weights']
        class_weights = torch.FloatTensor(c_weights).cuda()

        class_no = config['data']['class_no']
        class_names = config['data']['class_names']
        epochs = config['model']['epochs']
        learning_rate = config['model']['learning_rate']
        batch_s = config['model']['batch_size']
        vis_train_images = config['data']['visualize_images']['train']
        save_image_color_legend()

        # Optimizer
        if config['model']['optimizer'] == 'adam':
            optimizer = torch.optim.Adam([
                dict(params=model.parameters(), lr=learning_rate),
            ])
        elif config['model']['optimizer'] == 'sgd_mom':
            optimizer = torch.optim.SGD([
                dict(params=model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True),
            ])
        else:
            raise Exception('Choose valid optimizer!')

        min_trace = config['model']['min_trace']

        # Training loop
        for i in range(0, epochs):

            print('\nEpoch: {}'.format(i))
            model.train()
            train_loss = 0
            # Stop of the warm-up period
            if i == 5: #10 for cr_image_dice // 5 rest of the methods
                print("Minimize trace activated!")
                min_trace = True
                self.alpha = config['model']['alpha']
                print("Alpha updated", self.alpha)

            preds_train = []
            labels_train = []
            # Training in batches
            for j, (images, labels, imagename, ann_ids) in enumerate(trainloader):
                # Loading data to GPU
                images = images.cuda().float()
                labels = labels.cuda()
                if config['data']['prob']:
                    labels = labels.float()
        
          
                ann_ids = ann_ids.cuda().float()

                # zero the parameter gradients
                optimizer.zero_grad()

                if config['data']['ignore_last_class']!=-1:
                    ignore_index = int(config['data']['ignore_last_class'])  # deleted class is always set to the last index
                else:
                    ignore_index = -100  # this means no index ignored
                self.ignore_index = ignore_index


                y_pred = model(images)


                    
                if config['data']['prob']:
                    loss = torch.nn.CrossEntropyLoss(weight=class_weights,reduction='none')(
                        y_pred, labels)
      
                    loss = loss[labels[:, 0] != 1].mean()

                else:
                    loss = torch.nn.CrossEntropyLoss(weight=class_weights,ignore_index=ignore_index,reduction='mean')(
                        y_pred, labels)

                train_loss+=loss.item()
                y_pred_max = torch.argmax(y_pred, dim=1)
                preds_train.append(y_pred_max.cpu().detach().numpy().astype(np.int8).copy().flatten())
                if config['data']['prob']:
                    y_label_max = torch.argmax(labels, dim=1)
                else:
                    y_label_max = labels
                labels_train.append(y_label_max.cpu().detach().numpy().astype(np.int8).copy().flatten())
                # Backprop
                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()
                
            
                # Save results in training 
                if j % int(config['logging']['interval']) == 0:
                    print("Iter {}/{} - batch loss : {:.4f}".format(j, len(trainloader), loss))
                    if not config['data']['crowd']:
                        train_results = self.get_results(y_pred_max, y_label_max,class_names, class_no)
                        
                        log_results(train_results, mode='train', step=(i * len(trainloader) * batch_s + j))
                        for k in range(len(imagename)):
                            if imagename[k] in vis_train_images:
                                labels_save = labels[k].cpu().detach().numpy()
                                y_pred_max_save = y_pred_max[k].cpu().detach().numpy()
                                images_save = images[k]  # .cpu().detach().numpy()


            train_results = self.get_results(preds_train, labels_train,class_names, class_no)
            print('RESULTS for train')
            print(train_results)
            # Save validation results
            writer.add_scalar('train_loss', train_loss/len(trainloader), i+1)
            val_results,val_loss = self.evaluate(validateloader, writer,i,mode='val')             

            log_results(val_results, mode='val', step=int((i + 1) * len(trainloader) * batch_s))
      

            mlflow.log_metric('finished_epochs', i + 1, int((i + 1) * len(trainloader) * batch_s))

            # Save model
            metric_for_saving = val_results['macro_dice']
            writer.add_scalar(f'val_macro_dice', metric_for_saving, i+1)

            model_dir = 'models'
            dir = os.path.join(config['logging']['experiment_folder'], model_dir)
            os.makedirs(dir, exist_ok=True)
            out_path = os.path.join(dir, f'{i+1}.pth')
            torch.save(model, out_path)
            if max_score > val_loss:
                save_model(model)
                max_score = val_loss

            # LR decay
            if i > config['model']['lr_decay_after_epoch']:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / (1 + config['model']['lr_decay_param'])



        
    def test(self, testloader):
        save_image_color_legend()
        results = self.evaluate(testloader)
        log_results(results, mode='test', step=None)
        save_results(results)

    def evaluate(self, evaluatedata, writer,epoch,mode='test'):
        config = globals.config
        class_no = config['data']['val']['class_no']
        class_names = config['data']['val']['class_names']
        c_weights = config['data']['val']['class_weights']
        class_weights = torch.FloatTensor(c_weights).cuda()
        if mode=='test':
            print("Testing the best model")
            model_dir = 'models'
            dir = os.path.join(globals.config['logging']['experiment_folder'], model_dir)
            model_path = os.path.join(dir, 'best_model.pth')
            model = torch.load(model_path)
        else:
            model = self.model

        device = self.device
        model.eval()
        val_loss = 0
        labels = []
        preds = []

        with torch.no_grad():
            for j, (test_img, test_label, _, _) in enumerate(evaluatedata):
                test_img = test_img.to(device=device, dtype=torch.float32)
                test_pred = model(test_img)
          
                loss = torch.nn.CrossEntropyLoss(weight=class_weights,ignore_index=0,reduction='mean')(
                            test_pred, test_label.cuda())
   
                test_pred = torch.argmax(test_pred, dim=1)
                test_pred_np = test_pred.cpu().detach().numpy()
                test_label = test_label.cpu().detach().numpy()

                preds.append(test_pred_np.astype(np.int8).copy().flatten())
                labels.append(test_label.astype(np.int8).copy().flatten())
                
                val_loss+=loss.item()


            preds = np.concatenate(preds, axis=0, dtype=np.int8).flatten()
            labels = np.concatenate(labels, axis=0, dtype=np.int8).flatten()
            writer.add_scalar(f'{mode}_loss', val_loss/len(evaluatedata), epoch+1)
            results = self.get_results(preds, labels,class_names, class_no)

            print('RESULTS for ' + mode)
            print(results)
            return results,val_loss/len(evaluatedata)


    def get_results(self, pred, label,class_names, class_no):

        metrics_names = ['macro_dice', 'micro_dice', 'miou', 'accuracy']
        for class_id in range(1,class_no):
            metrics_names.append('dice_class_' + str(class_id) + '_' + class_names[class_id])
        if torch.is_tensor(pred):
            pred = pred.cpu().detach().numpy().copy().flatten()
        if torch.is_tensor(label):
            label = label.cpu().detach().numpy().copy().flatten()
        results = segmentation_scores(label, pred, metrics_names,class_names, class_no)

        return results
