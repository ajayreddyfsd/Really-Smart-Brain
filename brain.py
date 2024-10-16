from imageai.Classification import ImageClassification
import os

exec_path = os.getcwd()

'''
create instance
choose the model
set the model path 
load model into memory
'''


prediction = ImageClassification()
prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath(os.path.join(exec_path, 'mobilenet_v2-b0353104.pth'))
prediction.loadModel()


'''
house.jpg is an unlabelled image for the model
so, based on it's knowledge, the model give top 5 predictions
u can set no. to how many predctions u like
'''
predctions, probabilities = prediction.classifyImage(os.path.join(exec_path,'godzilla.jpg'), result_count=5)


# just printing the results
for eachPred, eachProb in zip(predctions, probabilities):
    print(f'{eachPred} : {eachProb}')
