#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/LU>

#include <sys/stat.h>
#include <sys/types.h>

#define N 30

using namespace std;
using namespace Eigen;

int minElement(double* array, size_t size);

void pseudo_inverse(MatrixXd *O, MatrixXd P);

void read_file(char *filename, int size2, double **data);

double func(double factor, double mental, int func_num);
double func2(double factor, double mental, int func_num);
double inv_norm(int num, double val);

typedef Matrix<double,N,10,RowMajor> Mat_phi;

void setPhi(double (&factor)[N][10],double mental[N], int emo_num, Mat_phi *Phi);
void get_para(double (&factor)[N][10], double (&signal)[N][4], double mental[N], MatrixXd *W1_para);
void f_mm(MatrixXd W1_para, double (&factor)[N][10], double mental[N], double *output);
//mental model
double fE(MatrixXd W1_para, double (&factor)[N][10], double mental[N], double sig_total[4]);

void out_tmppara(char *filename, MatrixXd W1_para);

void grad_descent(double M_org[N], double* M,MatrixXd W1_para,double (&factor)[N][10], double sig_total[4],int id_num,int moto_count);

int main(int argc, const char* argv[]){//(argv[0]=./estimate),argv[1]=usrname

  double *_factor[N],*_signal[N];
  double factor_size[N][10],signal_size[N][4];//データ数は１５個、要素数は７こ、
  double factor[N][10], signal[N][4];
  for(int i=0;i<N;i++) _factor[i] = factor_size[i];
  for(int i=0;i<N;i++) _signal[i] = signal_size[i];

  char factor_name[100];
  char signal_name[100];
  char face_name[100];
  char err_log[100];
  MatrixXd W1_para(10,4);

  int id_num=0;
  if(argc > 1){
    std::string id_input_str = argv[1];
    id_num=stoi(id_input_str);
    printf("id is %d\n",id_num);
  }

  const int set_num = N;
  std::cout<<"N: "<<N<<std::endl;

  sprintf(factor_name,"./data/%d/factor_before.csv",id_num);
  sprintf(signal_name,"./data/%d/signal_before.csv",id_num);
  sprintf(err_log,"./data/%d/error_phi.csv",id_num);

  std::ofstream err_ofs(err_log);

  read_file(factor_name, 10,_factor);
  read_file(signal_name, 4,_signal);
  //cout<<factor_name<<std::endl;

  memcpy(factor,*_factor,sizeof(factor));
  memcpy(signal,*_signal,sizeof(signal));

  double m_org[N];
  double mental[N];
  srand((unsigned) time(NULL));
  for(int ment_num=0;ment_num<N;ment_num++){
    m_org[ment_num]={0};
  }
  memcpy(mental,m_org,sizeof(m_org));

  double dEdM[N] = {0};

  //int step_i=0;
  int errflg=0;
  double err=10;

  std::cout << std::fixed;
  int count=0;

  char tmp_w_filename[50] = "weight_zero_trial00.csv";
  ofstream init_out(tmp_w_filename,ios::app);
  ofstream mental_out("mental_log_trial00.csv",ios::app);

  init_out<<id_num<<","<<count<<"==================="<<std::endl;

  while(errflg==0){
    std::cout<<"err"<<std::setprecision(10)<<err<<" count:"<<count<<", M:";
    mental_out<<"err"<<std::setprecision(10)<<err<<" count:"<<count<<", M:";

    for(int m_check=0;m_check<N;m_check++){
      std::cout<<std::setprecision(2)<<mental[m_check]<<", ";
      mental_out<<std::setprecision(2)<<mental[m_check]<<", ";
    }
    std::cout<<std::endl;
    mental_out<<std::endl;

    //MatrixXd W1_para(10,4);
    get_para(factor, signal, mental, &W1_para);

    double sig_total[4]={0};
    for(int emo_num=0;emo_num<4;emo_num++){
      for(int col_num1 = 0; col_num1<N; col_num1++){
        sig_total[emo_num] += signal[col_num1][emo_num];
      }
    }

    int count_mgrad=0;
    double E_pre =fE(W1_para,factor,m_org,sig_total);

    //grad_descent(m_org,mental,W1_para,factor,sig_total,id_num,t,count);
    grad_descent(m_org,mental,W1_para,factor,sig_total,id_num,count);

    //S誤を求める
    err = abs(fE(W1_para, factor, mental,sig_total) - E_pre);

    init_out<<id_num<<","<<count<<"==================="<<std::endl;
    out_tmppara(tmp_w_filename, W1_para);
    err_ofs<<err<<std::endl;
    std::cout<<id_num<<","<<count<<std::endl;

    count++;
    memcpy(m_org,mental,sizeof(mental));
    if(err<=0.0000001){
      errflg=1;
    }else if(count>=50000000){//KENSYU: loop count
      errflg=1;
    }
    printf("checkpoint err%f\n",err);
  }
  char hap[100],sup[100],ang[100],sad[100],face[100];
  sprintf(hap,"./data/%d/hap_weight.csv",id_num);
  sprintf(sup,"./data/%d/sup_weight.csv",id_num);
  sprintf(ang,"./data/%d/ang_weight.csv",id_num);
  sprintf(sad,"./data/%d/sad_weight.csv",id_num);

  ofstream log1(hap,ios::app);
  ofstream log2(sup,ios::app);
  ofstream log3(ang,ios::app);
  ofstream log4(sad,ios::app);

  for(int factor_num=0;factor_num<10;factor_num++){
    log1<<W1_para(factor_num,0)<<endl;
    log2<<W1_para(factor_num,1)<<endl;
    log3<<W1_para(factor_num,2)<<endl;
    log4<<W1_para(factor_num,3)<<endl;
  }
  out_tmppara(tmp_w_filename, W1_para);
  double total[4]={0};
  f_mm(W1_para, factor, mental, total);
}


void out_tmppara(char *filename, MatrixXd W1_para){
  ofstream out_para(filename,ios::app);
  for(int e_num=0;e_num<4;e_num++){
    for(int f_num=0;f_num<10;f_num++){
      out_para<<W1_para(f_num,e_num)<<",";
    }
    out_para<<std::endl;
  }
}

void read_file(char *filename, int size2, double **data){
    int i,j;
    double hoga;
    double hage;

    std::string data_f[size2];
    std::stringstream ss;

    std::string line;

    std::ifstream ifs;

    ifs.open(filename);
    j=0;
    while(getline(ifs,line)){
      std::stringstream ss(line);
      for(i=0;i<size2;i++){
        getline(ss,data_f[i],'\t');
        hoga=std::stod(data_f[i]);
        hage=std::stod(data_f[i]);
        data[j][i]=hage;
      }
      j++;
    }

}

void grad_descent(double M_org[N], double* M,MatrixXd W1_para,double (&factor)[N][10], double sig_total[4],int id_num,int moto_count){
  double dEdM[N]={0};
  double err=100;
  double Msize[N];
  int count=0;
  ofstream mental_out("mental_log_trial2.csv",ios::app);

  while(err>0.000001){
    double E_pre  = fE(W1_para, factor, M_org,sig_total);
    //std::cout<<"E_pre: "<<E_pre<<", ";
    for(int i=0;i<N;i++){
      double tmp_M = M_org[i];
      double E_cand[3] = {0};
      double delta[3] = {0};
      for(int dm = -1; dm<2;dm++){
        M[i] = tmp_M + dm*10;
        //E_cand[dm+1] = fE(M);
        E_cand[dm+1] = fE(W1_para, factor, M,sig_total);
        //std::cout<<"E_cand["<<dm+1<<"]: "<<E_cand[dm+1]<<std::endl;
        delta[dm+1] = (E_cand[dm+1] - E_pre);
      }
      int index = minElement(delta,3);
      dEdM[i] = -delta[index];
      M[i]=tmp_M;
    }
    err=0;
    for(int j=0;j<N;j++){
      M[j] = M_org[j]+dEdM[j];
      err+=(dEdM[j])*(dEdM[j])/2.0;
    }
    //std::cout<<std::setprecision(4)<<"err:"<<err<<std::endl;
    memcpy(M_org,M,sizeof(Msize));

    mental_out<<"gd_descent("<<id_num<<"-"<<"-"<<moto_count<<"),"<<std::setprecision(10)<<err<<","<<count<<",";
    for(int m_check=0;m_check<N;m_check++){
      //cout<<std::setprecision(2)<<M[m_check]<<", ";
      mental_out<<std::setprecision(2)<<M[m_check]<<", ";
    }

    //cout<<std::endl;
    mental_out<<std::endl;
    count++;
    if(count>10000){
      break;
    }

  }

  //cout<<std::endl;
}



int minElement(double* array, size_t size){
  int min = array[0];
  int index = 0;
  //std::cout<<"element check"<<std::endl;
  //std::cout<<array[0]<<",";
  for (size_t i = 1; i < size; ++i) {
  //  std::cout<<array[i]<<",";
    if (min < array[i]) {
      min = array[i];
      index = i;
    }
  }
  if((abs(array[0] - array[1])<0.0001) && (abs(array[2] - array[1])<0.0001))  index = 1;
  //std::cout<<"index: "<<index<<std::endl;
  return index;
}


void setPhi(double (&factor)[N][10],double mental[N], int emo_num, Mat_phi *Phi){

  Mat_phi Phi1;//計画行列のこと
  double m_conv = 0.0;


  int func_num[10][4]={{0,1,2,3},{4,5,6,7},{8,9,10,11},//もとのやつパート１
      {12,13,14,15},{16,17,18,19},{20,21,22,23},//7つの状況に対する予測関数、各情動4つ分ずつある
      {24,25,26,27},{28,29,30,31},{32,33,34,35},{36,37,38,39}};//もとのやつパート２

  for(int data_num=0;data_num<N;data_num++){
    m_conv = 9.98/(1.0 + exp(-0.5 * mental[data_num]))+0.01;
    for(int sit_num=0;sit_num<10;sit_num++){
      if(emo_num==0){
        Phi1(data_num,sit_num)=func(inv_norm(func_num[sit_num][emo_num],factor[data_num][sit_num]),m_conv,func_num[sit_num][emo_num]);//(1e+1);//因子から表情推定値を出す。mentalは基底関数のパラメータ
      }else if(emo_num==1){
        Phi1(data_num,sit_num)=func(inv_norm(func_num[sit_num][emo_num],factor[data_num][sit_num]),m_conv,func_num[sit_num][emo_num]);//(1e+1);
      }else if(emo_num==2){
        Phi1(data_num,sit_num)=func(inv_norm(func_num[sit_num][emo_num],factor[data_num][sit_num]),m_conv,func_num[sit_num][emo_num]);//(1e+1);
      }else if(emo_num==3){
        Phi1(data_num,sit_num)=func(inv_norm(func_num[sit_num][emo_num],factor[data_num][sit_num]),m_conv,func_num[sit_num][emo_num]);//(1e+1);
      }
      //多分ここは大丈夫。
    }
  }
  *Phi = Phi1;
}

void get_para(double (&factor)[N][10], double (&signal)[N][4], double mental[N], MatrixXd *W1_para){
  Mat_phi Phi1,Phi2,Phi3,Phi4;//計画行列のこと  //行優先の行列

  setPhi(factor, mental,0,&Phi1);
  setPhi(factor, mental,1,&Phi2);
  setPhi(factor, mental,2,&Phi3);
  setPhi(factor, mental,3,&Phi4);

  MatrixXd Sig_tar(N,4);//目標値tに相当します//最後の１列は全部１

  for(int d_num=0;d_num<N;d_num++){
    for(int e_num=0;e_num<4;e_num++){
      Sig_tar(d_num,e_num)=signal[d_num][e_num];
      //std::cout<<"sig out:" << signal[d_num][e_num]<<",";
    }
    //std::cout<<std::endl;
  }

  MatrixXd Phi1_t(10,N),Phi2_t(10,N),Phi3_t(10,N),Phi4_t(10,N);//Phi共有できないのは、Mの値によって式の形が変わるため
  MatrixXd PP1(10,10),PP2(10,10),PP3(10,10),PP4(10,10);
  MatrixXd PP1_inv(10,10),PP2_inv(10,10),PP3_inv(10,10),PP4_inv(10,10);

  PP1=Phi1.transpose()*Phi1;
  PP2=Phi2.transpose()*Phi2;
  PP3=Phi3.transpose()*Phi3;
  PP4=Phi4.transpose()*Phi4;

  pseudo_inverse(&PP1_inv,PP1);
  pseudo_inverse(&PP2_inv,PP2);
  pseudo_inverse(&PP3_inv,PP3);
  pseudo_inverse(&PP4_inv,PP4);

  W1_para->col(0)=PP1_inv*Phi1.transpose()*Sig_tar.col(0);//どの因子が表情表出に大きな影響を及ぼすかの重み
  W1_para->col(1)=PP2_inv*Phi2.transpose()*Sig_tar.col(1);
  W1_para->col(2)=PP3_inv*Phi3.transpose()*Sig_tar.col(2);
  W1_para->col(3)=PP4_inv*Phi4.transpose()*Sig_tar.col(3);

}

void f_mm(MatrixXd W1_para, double (&factor)[N][10], double mental[N], double *output){

  Mat_phi Phi1,Phi2,Phi3,Phi4;//計画行列のこと
  setPhi(factor, mental,0,&Phi1);
  setPhi(factor, mental,1,&Phi2);
  setPhi(factor, mental,2,&Phi3);
  setPhi(factor, mental,3,&Phi4);

  //make new file
  char sigtar_name[50];
  sprintf(sigtar_name, "signal_out_%d.csv",N);
  std::ofstream sigtar_out(sigtar_name,std::ios::out|std::ios::trunc);
  for(int col_num1=0;col_num1<N;col_num1++){
    for(int emo_num=0;emo_num<4;emo_num++){//まだ閉じてない
      for(int sit_num=0;sit_num<10;sit_num++){
        switch(emo_num){
          case 0:
            output[0] += W1_para(sit_num,emo_num)*Phi1(col_num1,sit_num);
            break;
          case 1:
            output[1] += W1_para(sit_num,emo_num)*Phi2(col_num1,sit_num);
            break;
          case 2:
            output[2] += W1_para(sit_num,emo_num)*Phi3(col_num1,sit_num);
            break;
          case 3:
            output[3] += W1_para(sit_num,emo_num)*Phi4(col_num1,sit_num);
            break;
        }
      }
      //print output[emo_num]
      sigtar_out<<output[emo_num];
      if(emo_num<3){
        sigtar_out<<",";
      }else if(emo_num==3){
        sigtar_out<<std::endl;
      }

    }
    //std::cout<<std::endl;
  }

}
//mental model

double fE(MatrixXd W1_para, double (&factor)[N][10], double mental[N], double sig_total[4]){
  double output=0;
  double total[4]={0};

  f_mm(W1_para, factor, mental, total);
  for(int emo_num=0;emo_num<4;emo_num++){
    output+= 0.1*total[emo_num]-sig_total[emo_num];
  }
  return output;
}
