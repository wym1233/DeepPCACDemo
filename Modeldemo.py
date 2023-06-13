from Model import *
import torch

B=7
N=512
Ns=64
x=torch.rand(B,3,N)
xyz=torch.rand(B,N,3)

print('x:',x.shape)

Enc=AttributeEncoder(Nin=N,Ns=Ns)
y,y_geo=Enc(x,xyz)
print('y:',y.shape)

y_hat=y
print('y_hat:',y_hat.shape)

Dec=AttributeDecoder(Nout=N,Ns=Ns)
x_hat=Dec(y_hat,xyz)
print('x_hat:',x_hat.shape)

print('------------------------')
HE=HyperEncoder(Ns=Ns)
z,z_geo=HE(y,y_geo)
print('z:',z.shape)

z_hat=z
print('z_hat:',z_hat.shape)


HC=HyperDecoder(Ns=Ns)
ave,std=HC(z_hat,y_geo)
print('ave:',ave.shape,'std:',std.shape)


