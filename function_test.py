from PIL import Image
import core.metrics as Metrics
path = "/data/diffusion_data/infer/infer_128_220901_030446/results/hr_save/0_100_hr.png"
img = Image.open(path).convert('RGB')
# mean = np.mean(split_scores)
mean  = Metrics.calculate_IS(img)
#print(len(split_scores))
print('IS is %.4f' % mean)