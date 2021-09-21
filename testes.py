from scipy import spatial
# List1 = [4, 47, 8, 3]
List2 = [3, 52, 12, 16]
List3=[1,2,3,4]
result = 1 - spatial.distance.cosine(List2, List3)
print(result)

