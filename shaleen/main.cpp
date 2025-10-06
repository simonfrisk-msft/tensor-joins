#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <string.h>
#include <iostream>
#include <algorithm>
#include <set>
#include <bitset>
#include <sstream>
#include <map>
#include <inttypes.h>
#include <sys/time.h>
#include <mutex>
#include <shared_mutex>
#include <time.h>
#ifdef __APPLE__
#include <unordered_map>
#include <unordered_set>
#else
#include <tr1/unordered_map>
#include <tr1/unordered_set>
#endif
#include <omp.h>
#define EIGEN_USE_MKL_ALL
#include "Eigen/Dense"
using namespace Eigen;
using namespace std;
#ifndef __APPLE__
using namespace tr1;
#endif

template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
  if ( !v.empty() ) {
    out << '[';
    std::copy (v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b\b]";
  }
  return out;
}

template <typename T>
std::ostream& operator<< (std::ostream& out, const std::set<T>& v) {
  if ( !v.empty() ) {
    out << '[';
    std::copy (v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b\b]";
  }
  return out;
}

vector<vector<int>> records;
long long int token_domain_size = 0;
vector<pair<int, int>> degree_token;
vector<pair<int, int>> degree_set;
vector<pair<long long int, long long int>> cdf_token;
vector<pair<long long int, long long int>> cdf_set;
long long int fulljoinsize = 0;
long long int n = 0;

void readFileClick();
void testdata();
pair<int, int> optimize();

pair<int, int> optimize() {
  cout << "full join size and relation size: " << fulljoinsize << " " << n << endl;
  if (fulljoinsize <= 10 * n)
    return make_pair(0, INT_MAX); // evaluate full join and de-dup.
  double out_pi = (sqrt(fulljoinsize)/n)*fulljoinsize; // estimate from HLL.
  cout << "projection output size estimate : " << out_pi << endl;
  int delta_1 = records.size();
  int delta_2 = n*(delta_1/out_pi);
  cout << "delta1 and delta2 initial : "<< delta_1 << " " << delta_2 << endl;
  long long int theavy = 0, tlight = fulljoinsize/1000, prev_tlight = LONG_MAX, prev_theavy = 0, prev_delta_1 = 0,
  prev_delta_2;
  long long int heavy_sets = 0, heavy_tokens = 0, index_set, index_token;

  for(int i = 0; i < 10000; i++) {
    prev_tlight = tlight; prev_theavy = theavy; prev_delta_1 = delta_1; prev_delta_2 = delta_2;
    delta_1 = delta_1 * 0.95;
    delta_2 = n*delta_1/out_pi;
    cout << "delta1 delta2 : " << delta_1 << " " << delta_2 << endl;
    std::vector<pair<int, int>>::iterator iter = lower_bound(degree_set.begin(), degree_set.end(), make_pair(delta_2, 0));
    index_set = iter - degree_set.begin();
    heavy_sets = degree_set.size() - (index_set);
    cout << "heavy sets : " << heavy_sets << endl;
    iter = lower_bound(degree_token.begin(), degree_token.end(), make_pair(delta_1, 0));
    index_token = iter - degree_token.begin();
    heavy_tokens = degree_token.size() - (index_token);
    cout << "heavy tokens : " << heavy_tokens << endl;
    cout << "light set and light token expansion cost : " << cdf_set[index_set].second << " " << cdf_token[index_token].second <<  endl;
    //estimate join time for light sets or lights tokens. constant 1000 chosen to ensure that cost is not a large integer.
    tlight = cdf_set[index_set].second/1000.0 + cdf_token[index_token].second/1000.0;
    //estimate join time for heavy sets and heavy tokens. constant 10000 is chosen since matrix multiplication is an
    // order of magnitude faster than normal join processing.
    theavy = heavy_sets*heavy_tokens*heavy_sets/10000.0;
    cout << "tlight theavy : " << tlight << " " << theavy << endl;
    cout << "prev_tlight prev_theavy : " << prev_tlight << " " << prev_theavy << endl;
    if((prev_tlight + prev_theavy)  <= (theavy + theavy) && prev_theavy > 0) {
      return make_pair(prev_delta_1, prev_delta_2);
    }
  }
  return make_pair(prev_delta_1, prev_delta_2);
}

int main(int argc, char ** argv) {
  srand(time(NULL));
  int c = 1;
  int num_threads = 1;
  omp_set_num_threads(num_threads);
  timeval starting, ending, s1, t1, s2, t2;

  testdata();

  cout << " number of sets : " << records.size() << endl;
  gettimeofday(&starting, NULL);

  vector<vector<int>> settotoken(records.size());
  vector<vector<int>> tokentoset(token_domain_size);
  for (int i = 0; i < records.size(); i++) {
    if (records[i].size() < c) continue;
    for (int j = 0; j < records[i].size(); j++)  {
      settotoken[i].push_back(records[i][j]);
      tokentoset[records[i][j]].push_back(i);
      n+=1;
    }
    degree_set.push_back(make_pair(records[i].size(), i));
  }

  int list_sizes = 0;
  for(int i = 0 ; i < settotoken.size(); i++) {
    for (auto j : settotoken[i])
      list_sizes += tokentoset[j].size();
    cdf_set.push_back(make_pair(settotoken[i].size(), list_sizes)); // store out_join of each set.
    list_sizes = 0;
  }

  for(int i = 0 ; i < tokentoset.size(); i++) {
    cdf_token.push_back(make_pair(tokentoset[i].size(), tokentoset[i].size()));
    degree_token.push_back(make_pair(tokentoset[i].size(), i));
    fulljoinsize += tokentoset[i].size()*tokentoset[i].size(); // assuming self join
  }

  sort(degree_set.begin(), degree_set.end(), [] (const pair<int, int> & a, const pair<int, int> & b) {
      return a.first < b.first;
  });
  sort(degree_token.begin(), degree_token.end(), [] (const pair< int, int> & a, const pair< int, int> & b) {
      return a.first < b.first;
  });


  sort(cdf_set.begin(), cdf_set.end(), [] (const pair<int, int> & a, const pair<int, int> & b) {
      return a.first < b.first;
  });
  sort(cdf_token.begin(), cdf_token.end(), [] (const pair<long long int, long long  int> & a, const pair<long long int, long long  int> & b) {
      return a.first < b.first;
  });



  // actually make cdf
  for (int i = 0 ; i < cdf_token.size(); i++) {
    if (i == 0)
      cdf_token.at(i).second = cdf_token.at(i).second*cdf_token.at(i).second;
    if (i > 0)
      cdf_token.at(i).second = cdf_token.at(i-1).second + cdf_token.at(i).second*cdf_token.at(i).second;
  }

  for (int i = 0 ; i < cdf_set.size(); i++) {
    if (i > 0)
      cdf_set.at(i).second += cdf_set.at(i-1).second;
  }

//  cout << "cdf_set : ";
//
//  for (auto i : cdf_set) {
//    cout << i.first << " " << i.second << ",";
//  }
//
//  cout << endl;
//
//  cout << "cdf_token : ";
//
//  for (auto i : cdf_token) {
//    cout << i.first << " " << i.second << ",";
//  }
//
//  cout << endl;

  pair<int, int> res = optimize();
  int delta_set = res.second; // res.second; debug by manually setting thresholds here
  int delta_token = res.first; // res.first; debug by manually setting thresholds here
  gettimeofday(&s1, NULL);

  cout << "Starting..." << endl;
  gettimeofday(&t1, NULL);
  vector<int> local_counter;
  for(int i = 0 ; i < num_threads; i++) {
    local_counter.push_back(0);
  }

  #pragma omp parallel
  {
    int threadnum = omp_get_thread_num(), numthreads = omp_get_num_threads();
    cout << "threadnum : " << threadnum << " numthreads : " << numthreads << endl;
    vector<int> dedup(records.size());
    int counter = 0;
    int low = records.size()*threadnum/numthreads, high = records.size()*(threadnum + 1)/numthreads;
    cout << low << " " << high << endl;
    for (int j = low; j < high; j++) {
      auto& k = settotoken[j];
      if (k.size() <= delta_set) {
        std::fill(dedup.begin(), dedup.end(), 0);
        for (auto token : k) {
          auto &l = tokentoset[token];
          for (auto& set : l) {
            if (!dedup[set]) {
              dedup[set]++;
              ++counter;
            }
          }
        }
      } else {
        std::fill(dedup.begin(), dedup.end(), 0);
        for (auto token : k) {
          auto &l = tokentoset[token];
          if (l.size() <= delta_token) {
            for (auto &set : l) {
              if (!dedup[set] && settotoken[set].size() <= delta_set) {
                dedup[set]++;
                ++counter;
              }
            }
          }
        }
      }
    }
    local_counter[threadnum] = counter;
  }

  gettimeofday(&s2, NULL);

  std::vector<pair<int, int>>::iterator iter = lower_bound(degree_token.begin(), degree_token.end(), make_pair(delta_token, 0));
  long long int index_token = iter - degree_token.begin();
  int heavy_tokens = degree_token.size() - (index_token);
  unordered_map<int,int> heavy_token_map(heavy_tokens);
  int counter = 0;
  for (int i = index_token; i < degree_token.size(); ++i) {
    heavy_token_map[degree_token[i].second] = counter++;
  }
  cout << "counter : " << counter << endl;
  iter = lower_bound(degree_set.begin(), degree_set.end(), make_pair(delta_set, 0));
  int index_set = iter - degree_set.begin();
  int heavy_sets = degree_set.size() - (index_set);
  MatrixXf A;
  A.resize(heavy_sets, heavy_token_map.size());
  A.setZero();
  int heavy_set_index = -1;
  for (int i = 0 ;i < records.size() ; ++i) {
    if (records[i].size() >= delta_set) {
      ++heavy_set_index; // records[i] is a heavy set
      for (auto& token : records[i]) {
        if (tokentoset[token].size() >= delta_token) {
          A.coeffRef(heavy_set_index, heavy_token_map[token]) = 1.0f;
        }
      }
    }
  }

  gettimeofday(&t2, NULL);

  MatrixXf B;
  B.noalias()=A*A.transpose();
  int heavy_output = 0;

  for (int i = 0; i < B.rows(); ++i) {
    for (int j = 0; j < B.cols(); ++j) {
      if (B.coeff(i,j) >= c) {
        ++heavy_output;
      }
    }
  }

  gettimeofday(&ending, NULL);
  cout << "Join Time: " << ending.tv_sec - t1.tv_sec + (ending.tv_usec - t1.tv_usec) / 1e6 << endl;
  cout << "  light sets and tokens: " << s2.tv_sec - t1.tv_sec + (s2.tv_usec - t1.tv_usec) / 1e6 << endl;
  cout << "  heavy sets: " << t2.tv_sec - s2.tv_sec + (t2.tv_usec - s2.tv_usec) / 1e6 << endl;
  cout << "All Time - Initial Time: " << ending.tv_sec - t1.tv_sec + (ending.tv_usec - t1.tv_usec) / 1e6 << endl;
  cout << "All: " << ending.tv_sec - starting.tv_sec + (ending.tv_usec - starting.tv_usec) / 1e6 << endl;
  cout << local_counter << endl;
  cout << heavy_output << endl;
  return 0;
}

void testdata() {
  set<int> domain_token;
  map<int, vector<int>> temp_map;
  for(int i = 0 ; i < 5000; i++) {
    for(int j = 0 ; j < 5000; j++) {
      if (rand() % 2  == 0) {
          temp_map[i].push_back(j);
          domain_token.insert(j);
      }
    }
  }
  for (auto&it : temp_map) {
    records.push_back(it.second);
  }
  token_domain_size = domain_token.size();
}

void readFileClick() {
  cout << "here";
  std::ifstream file("filelocation");
  set<int> domain_token;
  std::string   line;
  map<int, vector<int>> temp_map;
  int counter = 0;
  while(std::getline(file, line)) {
    std::stringstream  lineStream(line);
    int count; string papers;
    // Read an integer at a time from the line
    while(lineStream >> count >> papers)
    {
      std::vector<int> vect;
      std::stringstream ss(papers);
      int i;
      while (ss >> i) {
        vect.push_back(i);
        domain_token.insert(i);
        if (ss.peek() == ',')
          ss.ignore();
      }
      // Add the integers from a line to a 1D array (vector)
      records.push_back(vect);
      // cout << vect.size()<< endl;
    }
  }
  token_domain_size = domain_token.size();
}
