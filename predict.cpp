#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <stdlib.h>
#include <math.h>

#include <Eigen/Core>
#include <Eigen/LU>

#include <sys/stat.h>
#include <sys/types.h>
#define N 30


//#define ERROR 0.01;//1e-6;

using namespace std;
using namespace Eigen;


//森澤くんありがとうとしか言えない
/* OにPの擬似逆行列を代入するプログラム */
void pseudo_inverse(MatrixXd *O, MatrixXd P);

double func(double factor, double mental, int func_num);
double func2(double factor, double mental, int func_num);

typedef Matrix<double,N,10,RowMajor> Mat_phi;

//1〜10種類ある予測関数番号をもらうと計算をする関数を作ります
int main(int argc, const char* argv[]){//(argv[0]=./estimate),argv[1]=usrname

//これも最後のコマンド引数はユーザ名にする
	ifstream read_fact;//状況データを読み込む
	ifstream read_ment;//メンタル推定結果データを読み込む
	
	char mental_file_name[100];
	char para_name[100];
	int step;

	sprintf(mental_file_name,"./mental/%s.csv",argv[1]);//コマンドライン引数からのmentalの配列
	sprintf(para_name,"./para/%s_para.csv",argv[1]);//パラメタもinit毎に分類
  ifstream get_men(mental_file_name);
  //TODO

	string line;
	string line2;
	double factor[100][10];//データ数は１５個、要素数は７こ、
	double mental[100];
	double dM[N][5];//Sのでーた
	string data_f[10];
	string data_s[10];
	double hoga;
	double hage;
	int i;
	double tmp=0;
	int row=0;//全部で83ある
	stringstream ss;
	int j=0;//読み込んだ行数を覚えておく？じゃなくて、j=50になったらリセット

	char fact[100];
	sprintf(fact,"./factor/%s_factor.csv",argv[1]);//パラメタもinit毎に分類
	read_fact.open(fact);
	while(getline(read_fact,line)){
		stringstream ss(line);
		for(i=0;i<10;i++){
			getline(ss,data_f[i],',');
			hoga=stod(data_f[i]);
			hage=stod(data_f[i]);
			factor[j][i]=hage;
		}
		j++;
	}

	read_ment.open(mental_file_name);
//	double mental_sub[100][2];
	j=0;
  for(int men_num=0;men_num<N;men_num++){
    get_men>>mental[men_num];
  }
  typedef Matrix<double,N,11,RowMajor> Mat_phi;
	Mat_phi Phi1;//計画行列のこと

	MatrixXd Phi1_t(11,N);//Phi共有できないのは、Mの値によって式の形が変わるため
	MatrixXd PP1(11,11);
	MatrixXd PP1_inv(11,11);
	MatrixXd dm_tar(N,1);//計算した心的状態からdMを求めて、それを目標値とする
	MatrixXd W1_para(11,1);//各行のみ取り出せるんじゃないかしら

	int func_num[11]={0,1,2,3,4,5,6,7,8,9,10};

	for(int d_num=0;d_num<N;d_num++){
		dm_tar(d_num,0)=mental[d_num+1]-mental[d_num];
	}
	
	for(int step_num=0;step_num<N;step_num++){
		for(int data_num=0;data_num<step_num;data_num++){
			for(int sit_num=0;sit_num<10;sit_num++){
				Phi1(data_num,sit_num)=func2(factor[data_num][sit_num],mental[data_num],func_num[sit_num])/14.0;
			}
			Phi1(data_num,10)=func2(0,mental[data_num],func_num[10])/14.0;
		}
	}
	PP1=Phi1.transpose()*Phi1;
	pseudo_inverse(&PP1_inv,PP1);
	W1_para.col(0)=PP1_inv*Phi1.transpose()*dm_tar.col(0);//ここは列で指定したいのだが
	ofstream para(para_name);//Wパラメタをフォルダに出力します

	char dmname[100];
	sprintf(dmname,"./men/%s_para.csv",argv[1]);//パラメタもinit毎に分類
	ofstream dm_out(dmname);

	double sum_para=0;

	for(int c=0;c<11;c++){
		sum_para+=W1_para(c,0);
	}

	for(int c=0;c<11;c++){
		para<<10*W1_para(c,0)/sum_para<<endl;
		cout<<10*W1_para(c,0)/sum_para<<endl;
	}

	double delta_mental[N+1];
	double tmp_m=0;
	for(int step=0;step<N;step++){
		for(int a=0;a<step;a++){
			for(int b=0;b<11;b++){
				tmp_m+=W1_para(b,0)*Phi1(a,b)/14.0;
			}
			if(tmp_m<=-1){
					tmp_m=-1;
			}
			delta_mental[a+1]=tmp_m;
			tmp_m=0;
		}
	}
	for(int data=0;data<30;data++){
			dm_out<<delta_mental[data]<<std::endl;
	}

}
