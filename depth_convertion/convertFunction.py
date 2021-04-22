# convert depth value to distance 

# (for ERP projection)
# equation:
#   Depth value = vmax * ( (1/R) - (1/RFar) ) / ( (1/Rnear) - (1/Rfar) )
# para:
#   - depth: depth value
#   - RNear: min distance
#   - RFar: max distance
# retrun:
#   - R: real distance
def ERP_depth_convert(depth, RNear, RFar):
    vmax = 65535
    if(RFar == 'X'):
        R = 1 / ( (1/vmax) * ( depth * (1/RNear)  ) )
    else:
        R = 1 / ( (1/vmax) * ( depth * ( (1/RNear) - (1/RFar) ) ) + (1/RFar) )
    return R

# (for perspective projection)
# equation:
# Depth value = ( 1 << num_bits ) * ( a + b / z )
# a = zFar / ( zFar - zNear )
# b = ( zFar * zNear ) / ( zNear - zFar )
# para:
#   - depth : depth value
#   - num_bits: num of bits of Z precision
#   - zNear: given by dataset configuration
#   - zFar: given by dataset configuration
#  return:
#   - Z: real distance
def perspective_depth_convert(depth, num_bits, zNear, zFar):
    a = zFar / ( zFar - zNear )
    b = ( zFar * zNear ) / ( zNear - zFar )
    Z = ( pow(2,num_bits) * b ) / ( depth - pow(2,num_bits) * a)
    return Z