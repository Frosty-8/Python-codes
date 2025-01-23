# In this code we have a problem to set the entire 
# row and column to zero if an element in the matrix is zero.
# The approach is to use two sets to store the rows and columns
# that have zero elements. 

# Then we iterate through the matrix and if the row or column is in the set then we set the element
# to zero. This approach has a time complexity of O(m*n) and
# a space complexity of O(m+n) where m is the number of rows
# and n is the number of columns.


def set_zeroes(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    zero_rows = set()
    zero_cols = set()

    for i in range(rows):
        for j in range(cols):
            if matrix[i][j]==0:
                zero_rows.add(i)
                zero_cols.add(j)

    for i in range(rows):
        for j in range(cols):
            if i in zero_rows or j in zero_cols:
                matrix[i][j]=0
    
    return matrix
    
matrix = [[0,2,3,0],
          [4,1,7,8],
          [9,1,4,6]]
result = set_zeroes(matrix)
print(result)
