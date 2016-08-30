//����ָ�� -I ���include���Ŀ¼
//mex omp.cpp -I'E:\CppLibrary\eigen'
//h=omp(y,F,t);
#include <Eigen/Eigen>
#include "mex.h"
#include <stdio.h>
#include <vector>

using namespace Eigen;
using namespace std;

const int M = 360;			//ֻ�ܴ���һ�־����С��
const int N = 2520;			//
const int t = 84;			//��������
const double norm_v = 0.142857;	//2520ѡ360�Ĳ��ָ���Ҷ������й�һ��ֵ

VectorXcd rx_vcd(M);		//�����ź�
MatrixXcd F_mcd(M, N);	//���о���
VectorXcd r_n(M);		//�в�
MatrixXcd Aug(M, t);	//��������
int idxSet[t];			//�±꼯��
VectorXcd h_est(t);		//���Ƶ�h
//double norm_F[N];
vector<bool> idx_flag(N, false);

void printf_complex(double r, double i)
{
	printf("%lf", r);
	if (i < 0) printf("-");
	else printf("+");
	printf("i%lf, ", abs(i));
}
void printf_complex(complex<double> cd)
{
	printf("%lf", cd.real());
	if (cd.imag() < 0) printf("-");
	else printf("+");
	printf("i%lf, ", abs(cd.imag()));
}
/*
void norm_value()
{
//���ڲ��ָ���Ҷ���󣬸��й�һ��ֵ��ͬ
//2520ѡ360��ֵ��Ϊ 0.142857
for (int i = 0; i < N; i++)
{
double pd = 0.0;
for (int j = 0; j < M; j++)
{
//double tmp = abs(F_mcd(j, i));
pd += pow(abs(F_mcd(j, i)), 2);
}
norm_F[i] = pd;
}

for (int i = 0; i < N; i++)
{
printf("%lf, ", norm_F[i]);
}
printf("\n");
}
*/

int find_max_pos()
{
	int max_i = 0;
	double max_d = 0.0;
	double product_vec[N];
	for (int i = 0; i < N; i++)
	{
//		if (idx_flag[i]) continue;
		double pd = 0.0;
		complex<double> cd(0, 0);
		for (int j = 0; j < M; j++)
		{
			cd += conj(F_mcd(j, i))*r_n(j); //����
		}
		pd = abs(cd);
		//pd = pd*pd / norm_F[i];
		product_vec[i] = pd;
		if (pd > max_d)
		{
			max_d = pd;
			max_i = i;
		}
	}
	printf("max=%lf, ", max_d);
	return max_i;
}

void omp(int iter_th)
{
	r_n = rx_vcd;
	//norm_value();
	for (int it = 0; it < iter_th; it++)
	{
		int pos = find_max_pos();
		printf("pos = %d\n", pos);
		idxSet[it] = pos;
		idx_flag[it] = true;

		for (int i = 0; i < M; i++)
		{
			Aug(i, it) = F_mcd(i, pos);
			//F_mcd(i, pos) = 0;
		}
		MatrixXcd Aug_t(M, it + 1);	//��������
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j <= it; j++)
			{
				Aug_t(i, j) = Aug(i, j);
			}
		}

		//��С����
		auto t_Aug_t = Aug_t.conjugate().transpose();
		auto gm = t_Aug_t*Aug_t;
		auto inv_gm = gm.inverse();
		auto la = inv_gm*t_Aug_t;
		auto aug_y = la*rx_vcd;
		//���²в�
		r_n = rx_vcd - Aug_t*aug_y;
		//if (it == 0)
		//{
		//	for (int i = 0; i < M; i++)
		//		printf_complex(r_n(i));
		//	printf("\n");
		//}


		if (it == iter_th - 1)
		{
			h_est = aug_y;
		}
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	//dimision
	size_t row = mxGetM(prhs[1]);
	size_t col = mxGetN(prhs[1]);
	//printf("M=%d, N=%d\n", row, col);

	//rx - 
	double *rxSig_r = mxGetPr(prhs[0]);
	double *rxSig_i = mxGetPi(prhs[0]);
	//printf("M=%d, N=%d\n", mxGetM(prhs[0]), mxGetN(prhs[0]));

	//printf("rx:\n");
	for (int i = 0; i < row; i++)
	{
		//printf_complex(rxSig_r[i], rxSig_i[i]);
		complex<double> cd(rxSig_r[i], rxSig_i[i]);
		rx_vcd(i) = cd;
		//printf_complex(rx_vcd(i));
	}
	//printf("\n");

	//���о��� ��������
	double *F_r = mxGetPr(prhs[1]);
	double *F_i = mxGetPi(prhs[1]);

	//printf("F:\n");
	for (int i = 0; i < row; i++) //��
	{
		for (int j = 0; j < col; j++) //��
		{
			//printf_complex(F_r[i + row*j], F_i[i + row*j]);
			complex<double> cd(F_r[i + row*j], F_i[i + row*j]);
			F_mcd(i, j) = cd;
			//printf_complex(F_mcd(i, j));
		}
		//printf("\n");
	}
	//printf("\n");

	//��������
	double *iter_p = mxGetPr(prhs[2]);
	int iter_th = (int)*iter_p;
	//printf("iter times: %d\n", iter_th);

	//���
	plhs[0] = mxCreateDoubleMatrix(N, 1, mxCOMPLEX);
	double *res_r = mxGetPr(plhs[0]);
	double *res_i = mxGetPi(plhs[0]);

	omp(iter_th);

	//����ֵ
	for (int i = 0; i < t; i++)
	{
		int idx = idxSet[i];
		res_r[idx] = h_est[i].real();
		res_i[idx] = h_est[i].imag();
	}
}