from torchmetrics.text import CharErrorRate
#preds = ['Aunty, tēnā koe, i whakapūare nei i ngā kuaha o tēnei whare i te rā nei,']
#target = ['Aunty, tēnā koe, i whakapūare nei i ngā kuaha o tēnei whare i te rā nei,']
cer = CharErrorRate()

preds = []
target = []

path = '/home/ubuntu/whisper-vdi/data/'
model_name = 'large-v3-turbo'

gt = open(path + "paraini.txt", "rt")
gt = str(gt).lower()
preds = open(path + model_name + '.txt', "rt")
preds = str(preds).lower()

cer = CharErrorRate()
res = cer(preds, gt)
print(str(res.numpy()))

with open(path + 'transcription_times' + '.csv', 'a') as f:
    f.write(str(res.numpy()) + '\n')
f.close()