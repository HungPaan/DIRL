#include<iostream>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<map>
#include<algorithm>
#include<vector>
#include<cmath>
using namespace std;
namespace py = pybind11;
// ranked_user_label, np.array([top_k]), np.array([num_users]
py::array_t<double> gaotest(py::array_t<long long>& input1, py::array_t<long long>& input2,
py::array_t<long long>& input3)
{
    py::buffer_info buf1 = input1.request();
    py::buffer_info buf2 = input2.request();
    py::buffer_info buf3 = input3.request();

    auto ranked_user_label = input1.unchecked<2>();
    auto top_k_array = input2.unchecked<1>();
    auto num_users_array = input3.unchecked<1>();

    long long num_data = buf1.shape[0];
    long long top_k = top_k_array(0);
    long long num_users = num_users_array(0);

    double dcg[num_users] = {0.0};
    double idcg[num_users] = {0.0};
    double logsum[top_k+1] = {0.0};
    double ndcg = 0.0;
    double recall = 0.0;
    double precision = 0.0;

    long long rank_per_user[num_users] = {0};
    long long num_per_user_pos[num_users] = {0};

    double num_per_user_top_k[num_users] = {0.0};
    double norm = 0.0;

    for(long long j=1; j<=top_k; j++)
    {
       logsum[j] = logsum[j-1] + 1.0 / log2(j+1);
    }

    for(long long i=0; i<num_data; i++)
    {
        long long user = ranked_user_label(i, 0);
        long long flag = ranked_user_label(i, 1);
        rank_per_user[user]++;
        if (flag == 1)
        {
            num_per_user_pos[user]++;
            if (rank_per_user[user]<=top_k)
            {
                num_per_user_top_k[user]++;
                dcg[user] += 1.0/log2(rank_per_user[user]+1);
            }


        }

    }


    for(long long i=0; i<num_users; i++)
    {
        long long num_pos = num_per_user_pos[i];
        if(num_pos>=top_k)
        {
            idcg[i] = logsum[top_k];
        }
        else
        {
            idcg[i] = logsum[num_pos];
        }

        if(idcg[i]!=0)
        {

            ndcg += dcg[i] / idcg[i];
            recall += num_per_user_top_k[i] / num_per_user_pos[i];
            precision += num_per_user_top_k[i] / top_k;

            norm++;
        }


    }

    ndcg/=norm;
    recall/=norm;
    precision/=norm;

    auto result = py::array_t<double>(3);
    py::buffer_info resbuf = result.request();
    double* out = (double*)resbuf.ptr;
    out[0] = precision;
    out[1] = recall;
    out[2] = ndcg;


    return result;
}


PYBIND11_MODULE(ex, m) {

    m.doc() = "precision recall and ndcg!";
    m.def("gaotest", &gaotest);
}
/*
<%
setup_pybind11(cfg)
%>
*/