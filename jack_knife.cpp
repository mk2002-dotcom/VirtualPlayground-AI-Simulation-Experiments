// Jack knife error
#include <iostream>
#include <cmath>
#include<fstream>
#include<vector>

double func(double x){
	return pow(x, 2);
}


int main(){
	std::ifstream inputFile("input.txt");
	std::vector<double> data;
    double number;

	while (inputFile >> number){
		data.push_back(number);
	}

	inputFile.close();
	int data_size = data.size();
	std::cout << data_size << "data read" << std::endl;

    //Average
	double sum = 0e0;
	for (double num : data){
		sum += func(num);
	}
    std::cout << "Average:" <<  sum / (double)(data_size) << std::endl;

	//Error
	int bin_size = 1;
	int sample_num = (int) data_size / bin_size;
	double sample[sample_num];
	double sum_sample = 0e0;

    for (int i=0; i < sample_num; i++){
		double sum_jack;
		sum_jack = 0e0;

        for (int k=0; k < data_size; k++){
            if (k < bin_size * i || bin_size * (i+1) - 1 < k){
				sum_jack += func(data[k]);
			}
		}
		sample[i] = sum_jack / (double)(data_size - bin_size);
		sum_sample += sample[i];
	}

	double sample_aver = sum_sample / (double)(sample_num);
	double sum_error = 0e0;
    for (int i=0; i < sample_num; i++){
		sum_error += pow(sample[i] - sample_aver, 2);
	}

	std::cout << "Error:" << sqrt(sum_error * (double)(sample_num-1) / (double)(sample_num)) << std::endl;

	return 0;
}