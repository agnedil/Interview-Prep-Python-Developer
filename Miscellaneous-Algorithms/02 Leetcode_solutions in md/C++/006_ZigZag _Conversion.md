#6. ZigZag Conversion

**<font color=red>�Ѷ�:Medium</font>**

## ˢ������

> ԭ������

*https://leetcode.com/problems/zigzag-conversion/description/
* 
> ��������

```
The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)

P   A   H   N
A P L S I I G
Y   I   R
And then read line by line: "PAHNAPLSIIGYIR"

Write the code that will take a string and make this conversion given a number of rows:

string convert(string s, int numRows);
Example 1:

Input: s = "PAYPALISHIRING", numRows = 3
Output: "PAHNAPLSIIGYIR"
Example 2:

Input: s = "PAYPALISHIRING", numRows = 4
Output: "PINALSIGYAHRPI"
Explanation:

P     I    N
A   L S  I G
Y A   H R
P     I
```

## ���ⷽ��

> ˼·1
******- ʱ�临�Ӷ�: O(N)******- �ռ临�Ӷ�: O(N + numRows)******


������������Ŀ��˼��ʵ���ѣ�һ���˿��ܻῪһ����ά���飬Ȼ��Ͱ���Ŀ��˼���棬�������Ļ�ʱ�临�ӶȺͿռ临�Ӷȶ��Ƚϴ��������õķ�������һ�� string ���ͱ��� str ��resize ������� s ������ȣ�����ֻҪ�����ҵ� s[i] �� str �е�λ�ü���


```cpp
class Solution {
public:
    string convert(string s, int numRows) {
         string newStr;
    if(!s.length() || numRows == 1)
        return s;
    newStr.resize(s.length());
    int num = numRows * 2 - 2,col = s.length() / num,rem = (s.length() - 1) % num;
    vector<int> rowNum;
    for(int i = 0;i < numRows;++i)
        if(!i)
            s.length() % num ? rowNum.push_back(col + 1) : rowNum.push_back(col);
        else
        {
            if(i == numRows - 1)
                rem >= i ? rowNum.push_back(rowNum[i - 1] + (s.length() - 1) / num + 1) : rowNum.push_back(rowNum[i - 1] + (s.length() - 1) / num);
            else
            {
                int temp = 2 * numRows - i - 2,col1 = (s.length() - 1) / num;
                if(rem >= temp)
                    rowNum.push_back(rowNum[i - 1] + (col1 + 1) * 2);
                else if(rem >= i)
                    rowNum.push_back(rowNum[i - 1] + col1 * 2 + 1);
                else
                    rowNum.push_back(rowNum[i - 1] + col1 * 2);
            }
        }
    for(int i = 0;i < s.length();++i)
    {
        int index1 = i % num;
        int index2 = i / num;
        if(!index1)
            newStr[index2] = s[i];
        else if(index1 == numRows - 1)
            newStr[index2 + rowNum[index1 - 1]] = s[i];
        else if(index1 < numRows)
            newStr[index2 * 2 + rowNum[index1 - 1]] = s[i];
        else
        {
            int index3 = 2 * numRows - index1 - 2;
            newStr[index2 * 2 + 1 + rowNum[index3 - 1]] = s[i];
        }
    }
    return newStr;
    }
};
```