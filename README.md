import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

i1, i2 = 0.05, 0.1

w1, w2, w3, w4 = 0.15, 0.20, 0.25, 0.30
w5, w6, w7, w8 = 0.40, 0.45, 0.50, 0.55
b1, b2 = 0.35, 0.60
target_o1, target_o2 = 0.01, 0.99
eta = 0.5

net_h1 = w1 * i1 + w2 * i2 + b1
out_h1 = sigmoid(net_h1)

net_h2 = w3 * i1 + w4 * i2 + b1
out_h2 = sigmoid(net_h2)

net_o1 = w5 * out_h1 + w6 * out_h2 + b2
out_o1 = sigmoid(net_o1)

net_o2 = w7 * out_h1 + w8 * out_h2 + b2
out_o2 = sigmoid(net_o2)

E_total_o1 = 0.5 * (target_o1 - out_o1) ** 2
E_total_o2 = 0.5 * (target_o2 - out_o2) ** 2
E_total = E_total_o1 + E_total_o2

print("Forward Pass Values:")
print(f"out_h1: {out_h1}, out_h2: {out_h2}")
print(f"out_o1: {out_o1}, out_o2: {out_o2}")
print(f"Total Error: {E_total}")

d_E_total_out_o1 = -(target_o1 - out_o1)
d_E_total_out_o2 = -(target_o2 - out_o2)

d_out_o1_net_o1 = sigmoid_derivative(out_o1)
d_out_o2_net_o2 = sigmoid_derivative(out_o2)

d_net_o1_w5 = out_h1
d_net_o1_w6 = out_h2
d_net_o2_w7 = out_h1
d_net_o2_w8 = out_h2

d_E_total_w5 = d_E_total_out_o1 * d_out_o1_net_o1 * d_net_o1_w5
d_E_total_w6 = d_E_total_out_o1 * d_out_o1_net_o1 * d_net_o1_w6
d_E_total_w7 = d_E_total_out_o2 * d_out_o2_net_o2 * d_net_o2_w7
d_E_total_w8 = d_E_total_out_o2 * d_out_o2_net_o2 * d_net_o2_w8

w5_new = w5 - eta * d_E_total_w5
w6_new = w6 - eta * d_E_total_w6
w7_new = w7 - eta * d_E_total_w7
w8_new = w8 - eta * d_E_total_w8

d_out_h1_net_h1 = sigmoid_derivative(out_h1)
d_out_h2_net_h2 = sigmoid_derivative(out_h2)

d_net_h1_w1 = i1
d_net_h1_w2 = i2
d_net_h2_w3 = i1
d_net_h2_w4 = i2

d_E_total_w1= ((d_E_total_out_o1*d_out_o1_net_o1 * w5) + (d_E_total_out_o2*d_out_o2_net_o2 * w7))*d_out_h1_net_h1 * d_net_h1_w1
d_E_total_w2= ((d_E_total_out_o1*d_out_o1_net_o1 * w5) + (d_E_total_out_o2*d_out_o2_net_o2 * w7))*d_out_h1_net_h1 * d_net_h1_w2
d_E_total_w3= ((d_E_total_out_o1*d_out_o1_net_o1 * w6) + (d_E_total_out_o2*d_out_o2_net_o2 * w8))*d_out_h2_net_h2 * d_net_h2_w3
d_E_total_w4= ((d_E_total_out_o1*d_out_o1_net_o1 * w6) + (d_E_total_out_o2*d_out_o2_net_o2 * w8))*d_out_h2_net_h2 * d_net_h2_w4

w1_new = w1 - eta * d_E_total_w1
w2_new = w2 - eta * d_E_total_w2
w3_new = w3 - eta * d_E_total_w3
w4_new = w4 - eta * d_E_total_w4

w1, w2, w3, w4 = w1_new, w2_new, w3_new, w4_new
w5, w6, w7, w8 = w5_new, w6_new, w7_new, w8_new

print("\nUpdated Weights After Backward Pass:")
print(f"w1: {w1}, w2: {w2}, w3: {w3}, w4: {w4}")
print(f"w5: {w5}, w6: {w6}, w7: {w7}, w8: {w8}")
