# include <iostream>
# include <vector>

using namespace std;

int main(){
    int n = 3;
    vector<vector<int>> v_2;
    for (int i=0;i<n;i++){
        vector<int> tmp;
        for (int j=0;j<n;j++){
            tmp.push_back(i*j);
        }
        v_2.push_back(tmp);
    }
    int a_2[n][n];
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            a_2[i][j] = i*j;
        }
    }
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            cout<<"v_2:"<<&(v_2[i][j])<<" a_2:"<<&(a_2[i][j])<<endl;
        }
    }
    return 0;
}