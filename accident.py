import cv2
import tensorflow as tf
#rnn = tf.nn.rnn
#rnn_cell = tf.nn.rnn_cell
import argparse
import numpy as np
import os
import pdb
import time
import matplotlib.pyplot as plt
import sys

############### Global Parameters ###############
# path
train_path = '/media/HDD2/corgi/Conference/ACCV2016/dataset/train/'
test_path = '/media/HDD2/corgi/Conference/ACCV2016/dataset/test/'
demo_path = './features/'
default_model_path = './model/demo_model'
save_path = './model/'
image_path = './images/'
# batch_number
train_num = 126
test_num = 46


############## Train Parameters #################

# Parameters
learning_rate = 0.0001
n_epochs = 100
batch_size = 10
display_step = 10

# Network Parameters
n_input = 4096 # fc6 or fc7(1*4096)
n_detection = 21 # number of object of each image (include image features)
n_hidden = 512 # hidden layer num of LSTM
n_img_hidden = 256 # embedding image features 
n_att_hidden = 256 # embedding object features
n_classes = 2 # has accident or not
n_frames = 100 # number of frame in each video 
##################################################

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='accident_LSTM')
    parser.add_argument('--mode',dest = 'mode',help='train or test',default = 'demo')
    parser.add_argument('--model',dest = 'model',default= default_model_path)
    parser.add_argument('--gpu',dest = 'gpu',default= '0')
    args = parser.parse_args()

    return args


def build_model():

    # tf Graph input
    x = tf.placeholder("float", [None, n_frames ,n_detection, n_input])
    y = tf.placeholder("float", [None, n_classes])
    keep = tf.placeholder("float",[None])

    # Define weights
    weights = {
        'em_obj': tf.Variable(tf.random_normal([n_input,n_att_hidden], mean=0.0, stddev=0.01)),
        'em_img': tf.Variable(tf.random_normal([n_input,n_img_hidden], mean=0.0, stddev=0.01)),
        'att_w': tf.Variable(tf.random_normal([n_att_hidden, 1], mean=0.0, stddev=0.01)),
        'att_wa': tf.Variable(tf.random_normal([n_hidden, n_att_hidden], mean=0.0, stddev=0.01)),
        'att_ua': tf.Variable(tf.random_normal([n_att_hidden, n_att_hidden], mean=0.0, stddev=0.01)),
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=0.0, stddev=0.01))
    }
    biases = {
        'em_obj': tf.Variable(tf.random_normal([n_att_hidden], mean=0.0, stddev=0.01)),
        'em_img': tf.Variable(tf.random_normal([n_img_hidden], mean=0.0, stddev=0.01)),
        'att_ba': tf.Variable(tf.zeros([n_att_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes], mean=0.0, stddev=0.01))
    }

    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden,n_hidden,initializer= tf.random_normal_initializer(mean=0.0,stddev=0.01),use_peepholes = True,state_is_tuple = False)
    # using dropout in output of LSTM
    lstm_cell_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=1 - keep[0])
    # init LSTM parameters
    istate = tf.zeros([batch_size, lstm_cell.state_size])
    h_prev = tf.zeros([batch_size, n_hidden])
    # init loss 
    loss = 0.0  
    # Start creat graph
    for i in range(n_frames):
        # input features (Faster-RCNN fc7)
        X = tf.transpose(x[:,i,:,:], [1, 0, 2])  # permute n_steps and batch_size (n x b x h)
        # frame embedded
        image = tf.matmul(X[0,:,:],weights['em_img']) + biases['em_img'] # 1 x b x h
        # object embedded
        n_object = tf.reshape(X[1:n_detection,:,:], [-1, n_input]) # (n_steps*batch_size, n_input)
        n_object = tf.matmul(n_object, weights['em_obj']) + biases['em_obj'] # (n x b) x h
        n_object = tf.reshape(n_object,[n_detection-1,batch_size,n_att_hidden]) # n-1 x b x h

        # object attention
        brcst_w = tf.tile(tf.expand_dims(weights['att_w'], 0), [n_detection-1,1,1]) # n x h x 1
        image_part = tf.batch_matmul(n_object, tf.tile(tf.expand_dims(weights['att_ua'], 0), [n_detection-1,1,1])) + biases['att_ba'] # n x b x h
        e = tf.tanh(tf.matmul(h_prev,weights['att_wa'])+image_part) # n x b x h
        # softmax 
        e = tf.exp(tf.reduce_sum(tf.batch_matmul(e,brcst_w),2)) # n x b
        denomin = tf.reduce_sum(e,0) # b
        # the probability of each object
        alphas = tf.tile(tf.expand_dims(tf.div(e,denomin),2),[1,1,n_att_hidden]) # n x b x h
        # weighting sum
        attention_list = tf.mul(alphas,n_object) # n x b x h
        attention = tf.reduce_sum(attention_list,0) # b x h
        # concat frame & object
        fusion = tf.concat(1,[image,attention])
        # reuse variables
        if i > 0 :  tf.get_variable_scope().reuse_variables()       
        with tf.variable_scope("LSTM") as vs:
            outputs,istate = lstm_cell_dropout(fusion,istate)
            lstm_variables = [v for v in tf.global_variables() if v.name.startswith(vs.name)]
        # save prev hidden state of LSTM
        h_prev = outputs
        # FC to output
        pred = tf.matmul(outputs,weights['out']) + biases['out'] # b x n_classes
        # save the predict of each time step
        if i == 0:
            soft_pred = tf.reshape(tf.gather(tf.transpose(tf.nn.softmax(pred),(1,0)),1),(batch_size,1))
            all_alphas = tf.expand_dims(tf.div(e,denomin),0)
        else:
            temp_soft_pred = tf.reshape(tf.gather(tf.transpose(tf.nn.softmax(pred),(1,0)),1),(batch_size,1))
            soft_pred = tf.concat(1,[soft_pred,temp_soft_pred])
            temp_alphas = tf.expand_dims(tf.div(e,denomin),0)
            all_alphas = tf.concat(0,[all_alphas, temp_alphas])

        # positive example (exp_loss)
        pos_loss = -tf.mul(tf.exp(-(n_frames-i-1)/20.0),-tf.nn.softmax_cross_entropy_with_logits(pred, y))
        # negative example
        neg_loss = tf.nn.softmax_cross_entropy_with_logits(pred, y) # Softmax loss

        temp_loss = tf.reduce_mean(tf.add(tf.mul(pos_loss,y[:,1]),tf.mul(neg_loss,y[:,0])))
        #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        loss = tf.add(loss, temp_loss)
        
    # Define loss and optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss) # Adam Optimizer

    return x,keep,y,optimizer,loss,lstm_variables,soft_pred,all_alphas

def train():
    # build model
    x,keep,y,optimizer,loss,lstm_variables,soft_pred,all_alphas = build_model()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    # mkdir folder for saving model
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)
    # Initializing the variables
    init = tf.global_variables_initializer()
    # Launch the graph
    sess.run(init)
    saver = tf.train.Saver(max_to_keep=100)
    # Keep training until reach max iterations
    # start training
    for epoch in range(n_epochs):
         # random chose batch.npz
         epoch_loss = np.zeros((train_num,1),dtype = float)
         n_batchs = np.arange(1,train_num+1)
         np.random.shuffle(n_batchs)
         tStart_epoch = time.time()
         for batch in range(len(n_batchs)):
             batch_data = np.load(train_path+'batch_'+str(n_batchs[batch])+'.npz')
             batch_xs = batch_data['data']
             batch_ys = batch_data['labels']
             _,batch_loss = sess.run([optimizer,loss], feed_dict={x: batch_xs, y: batch_ys, keep: [0.5]})
             epoch_loss[batch] = batch_loss/batch_size
         # print one epoch
         print "Epoch:", epoch+1, " done. Loss:", np.mean(epoch_loss)
         tStop_epoch = time.time()
         print "Epoch Time Cost:", round(tStop_epoch - tStart_epoch,2), "s"
         sys.stdout.flush()
         if (epoch+1) %10 == 0:
            saver.save(sess,save_path+"model", global_step = epoch+1)
            print "Training"
            test_all(sess,train_num,train_path,x,keep,y,loss,lstm_variables,soft_pred)
            print "Testing"
            test_all(sess,test_num,test_path,x,keep,y,loss,lstm_variables,soft_pred)
    print "Optimization Finished!"
    saver.save(sess, save_path+"final_model")

def test_all(sess,num,path,x,keep,y,loss,lstm_variables,soft_pred):
    total_loss = 0.0

    for num_batch in range(1,num+1):
         # load test_data
         test_all_data = np.load(path+'batch_'+str(num_batch)+'.npz')
         test_data = test_all_data['data']
         test_labels = test_all_data['labels']
         [temp_loss,pred] = sess.run([loss,soft_pred], feed_dict={x: test_data, y: test_labels, keep: [0.0]})
         
         total_loss += temp_loss/batch_size

         if num_batch <= 1:
             all_pred = pred[:,0:90]
             all_labels = np.reshape(test_labels[:,1],[batch_size,1])
         else:
             all_pred = np.vstack((all_pred,pred[:,0:90]))
             all_labels = np.vstack((all_labels,np.reshape(test_labels[:,1],[batch_size,1])))

    evaluation(all_pred,all_labels)

    
def evaluation(all_pred,all_labels, total_time = 90, vis = False, length = None):
    ### input: all_pred (N x total_time) , all_label (N,)
    ### where N = number of videos, fps = 20 , time of accident = total_time
    ### output: AP & Time to Accident

    if length is not None:
        all_pred_tmp = np.zeros(all_pred.shape)
        for idx, vid in enumerate(length):
                all_pred_tmp[idx,total_time-vid:] = all_pred[idx,total_time-vid:]
        all_pred = np.array(all_pred_tmp)
        temp_shape = sum(length)
    else:
        length = [total_time] * all_pred.shape[0]
        temp_shape = all_pred.shape[0]*total_time
    Precision = np.zeros((temp_shape))
    Recall = np.zeros((temp_shape))
    Time = np.zeros((temp_shape))
    cnt = 0
    AP = 0.0
    for Th in sorted(all_pred.flatten()):
        if length is not None and Th == 0:
                continue
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0
        for i in range(len(all_pred)):
            tp =  np.where(all_pred[i]*all_labels[i]>=Th)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                time += tp[0][0] / float(length[i])
                counter = counter+1
            Tp_Fp += float(len(np.where(all_pred[i]>=Th)[0])>0)
        if Tp_Fp == 0:
            Precision[cnt] = np.nan
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) ==0:
            Recall[cnt] = np.nan
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            Time[cnt] = np.nan
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1

    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    _,rep_index = np.unique(Recall,return_index=1)
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])

    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    new_Time = new_Time[-np.isnan(new_Precision)]
    new_Recall = new_Recall[-np.isnan(new_Precision)]
    new_Precision = new_Precision[-np.isnan(new_Precision)]

    if new_Recall[0] != 0:
        AP += new_Precision[0]*(new_Recall[0]-0)
    for i in range(1,len(new_Precision)):
        AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

    print "Average Precision= " + "{:.4f}".format(AP) + " ,mean Time to accident= " +"{:.4}".format(np.mean(new_Time))
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    print "Recall@80%, Time to accident= " +"{:.4}".format(sort_time[np.argmin(np.abs(sort_recall-0.8))])

    ### visualize

    if vis:
        plt.plot(new_Recall, new_Precision, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall example: AUC={0:0.2f}'.format(AP))
        plt.show()
        plt.clf()
        plt.plot(new_Recall, new_Time, label='Recall-mean_time curve')
        plt.xlabel('Recall')
        plt.ylabel('time')
        plt.ylim([0.0, 5])
        plt.xlim([0.0, 1.0])
        plt.title('Recall-mean_time' )
        plt.show()


def vis(model_path):
    # build model
    x,keep,y,optimizer,loss,lstm_variables,soft_pred,all_alphas = build_model()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    # restore model
    saver.restore(sess, model_path)
    # load data
    all_data = np.load(demo_path+'demo.npz')
    data = all_data['data']
    labels = all_data['labels']
    det = all_data['det']
    ID = all_data['ID']
    # run result
    [all_loss,pred,weight] = sess.run([loss,soft_pred,all_alphas], feed_dict={x: data, y: labels, keep: [0.0]})
    folder_list = sorted(os.listdir(image_path))
    for i in range(len(ID)):
        plt.figure(figsize=(14,5))
        plt.plot(pred[i,0:90],linewidth=3.0)
        plt.ylim(0, 1)
        plt.ylabel('Probability')
        plt.xlabel('Frame')
        plt.show()
        folder_name = ID[i]
        bboxes = det[i]
        new_weight = weight[:,:,i]*255
        counter = 0 
        for img in sorted(os.listdir(image_path+folder_name)):
            frame = cv2.imread(image_path+folder_name+'/'+img)
            attention_frame = np.zeros((frame.shape[0],frame.shape[1]),dtype = np.uint8)
            now_weight = new_weight[counter,:]
            new_bboxes = bboxes[counter,:,:]
            index = np.argsort(now_weight)
            for num_box in index:
                if now_weight[num_box]/255.0>0.4:
                    cv2.rectangle(frame,(new_bboxes[num_box,0],new_bboxes[num_box,1]),(new_bboxes[num_box,2],new_bboxes[num_box,3]),(0,255,0),3)
                else:
                    cv2.rectangle(frame,(new_bboxes[num_box,0],new_bboxes[num_box,1]),(new_bboxes[num_box,2],new_bboxes[num_box,3]),(255,0,0),2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,str(round(now_weight[num_box]/255.0*10000)/10000),(new_bboxes[num_box,0],new_bboxes[num_box,1]), font, 0.5,(0,0,255),1,cv2.CV_AA)
                attention_frame[int(new_bboxes[num_box,1]):int(new_bboxes[num_box,3]),int(new_bboxes[num_box,0]):int(new_bboxes[num_box,2])] = now_weight[num_box]

            attention_frame = cv2.applyColorMap(attention_frame, cv2.COLORMAP_HOT)
            dst = cv2.addWeighted(frame,0.6,attention_frame,0.4,0)
            cv2.putText(dst,str(counter+1),(10,30), font, 1,(255,255,255),3)
            cv2.imshow('result',dst)
            c = cv2.waitKey(50)
            if c == ord('q') and c == 27:
                break;
            counter += 1
        
        cv2.destroyAllWindows()



def test(model_path):
    # load model
    x,keep,y,optimizer,loss,lstm_variables,soft_pred,all_alphas = build_model()
    # inistal Session
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    print "model restore!!!"
    print "Training"
    test_all(sess,train_num,train_path,x,keep,y,loss,lstm_variables,soft_pred)
    print "Testing"
    test_all(sess,test_num,test_path,x,keep,y,loss,lstm_variables,soft_pred)



if __name__ == '__main__':
    args = parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

    if args.mode == 'train':
           train()
    elif args.mode == 'test':
           test(args.model)
    elif args.mode == 'demo':
           vis(args.model)
