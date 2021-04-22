from inputYUV import * 
from convertFunction import * 

sampleNum = 5000

# ERP exp.
# dataset: ClassroomVideo
RNear = 0.8
RFar = 'X'#infinite
ERPSampleDepth = ERPSample(sampleNum)
ERPDistance = ERP_depth_convert(ERPSampleDepth, RNear, RFar)
print("- ERP")
print(f"depth value: {ERPSampleDepth}")
print(f"distance: {ERPDistance}")
# Perspective exp.
# dataset: TechnicolorPainter
zNear = 0.3
ZFar = 1.62
num_bits = 16
PerspectiveSampleDepth = PerspectiveSample(sampleNum)
perspectiveDistance = perspective_depth_convert(PerspectiveSampleDepth, num_bits, zNear, ZFar)
print("- Perspective")
print(f"depth value: {PerspectiveSampleDepth}")
print(f"distance: {perspectiveDistance}")