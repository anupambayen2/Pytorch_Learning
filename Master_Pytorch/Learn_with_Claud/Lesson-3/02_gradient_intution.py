import torch

x = torch.tensor([3.0], requires_grad=True)
f = x **2

print("When x=", x.item())
print("f(x) =x2 =", f.item())
print()

#Calculate gradient
f.backward()
print("Gradient (slope) at x=3:", x.grad.item())
print()

print("What does gradient=6 mean?")
print("If I increase x from 3.0 to 3.1 (tine increase of 0.1):")
print("f will increase by approximately: 0.1x6 = 0.6")
print()

# Let's verify!
x_new = 3.1
f_new = x_new ** 2
f_old = 3.0 **2
actual_change = f_new - f_old
predicted_change = 0.1 * 6

print("Actual Change:", actual_change)
print("Predicted Change", predicted_change)
print("Pretty Close")