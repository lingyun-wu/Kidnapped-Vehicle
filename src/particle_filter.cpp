/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    
    // Random number distribution setup
    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    int m = 100;                // number of particles
    num_particles = m;
    is_initialized = true;      // initialized

    // reset particle and weight arrays
    particles.resize(m);
    weights.resize(m, 1.0);

    // Initialize all particles' positions and weights
    for (int i = 0; i < m; ++i) {
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
        particles[i].id = i;
        particles[i].weight = 1.0;
    }

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    
    // Random number distribution
    double std_x = std_pos[0];
    double std_y = std_pos[1];
    double std_theta = std_pos[2];
    normal_distribution<double> dist_x(0, std_x);
    normal_distribution<double> dist_y(0, std_y);
    normal_distribution<double> dist_theta(0, std_theta);

    for (int i = 0; i < num_particles; ++i) {
        double theta = particles[i].theta;       // particle's current theta

        if (abs(yaw_rate) < 0.00001) {           // when yaw rate is very small, take it as straight moving
            particles[i].x += velocity * delta_t * cos(theta);
            particles[i].y += velocity * delta_t * sin(theta);
        } else {                                 // moving model
            particles[i].x += velocity / yaw_rate * (sin(theta+yaw_rate*delta_t) - sin(theta));
            particles[i].y += velocity / yaw_rate * (cos(theta) - cos(theta+yaw_rate*delta_t));
            particles[i].theta += yaw_rate * delta_t;
        }

        // add noise
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    
    // size constants
    int nP = predicted.size();
    int nO = observations.size();
    
    // Loop through observations vector
    for (int i = 0; i < nO; ++i) {
        double min_dist = numeric_limits<double>::max();     // minimum distance 
        int map_id = -1;                                     // id of predicted Landmark

        // Loop through predicted vector
        for (int j = 0; j < nP; ++j) {
            double x_dist = observations[i].x - predicted[j].x;
            double y_dist = observations[i].y - predicted[j].y;
            double distance = x_dist * x_dist + y_dist * y_dist;    // distance between predicted and observed landmarks

            // Upate the minimum distance if the new one is smaller
            if (distance < min_dist) {
                min_dist = distance;
                map_id = predicted[j].id;
            }
        }

        // Update the closest predicted measurement id to the observation measurement
        observations[i].id = map_id;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    
    // standard deviations of observations
    double sigma_x = std_landmark[0];
    double sigma_y = std_landmark[1];

    for (int i = 0; i < num_particles; ++i) {
        // particles position
        double x = particles[i].x;
        double y = particles[i].y;
        double theta = particles[i].theta;

        double sensor_range_square = sensor_range * sensor_range;   // sensor's range square
        vector<LandmarkObs> map_in_range;                           // array which contians the landmarks in the sensor range
        int cnt = 0;                   // count the index for elements in map_in_range
        // get the landmarks which are in the range
        for (int j = 0; j < int(map_landmarks.landmark_list.size()); ++j) {
            double x_map = map_landmarks.landmark_list[j].x_f;
            double y_map = map_landmarks.landmark_list[j].y_f;
            double dx = x - x_map;
            double dy = y - y_map;
            // landmarks in the sensor range
            if (dx*dx+dy*dy < sensor_range_square) {
                map_in_range.push_back(LandmarkObs{cnt, x_map, y_map});
                ++cnt;
            }
        }
        
        // transform the observation landmarks' coordinates to the map coordinates using homogeneous trnasformation
        vector<LandmarkObs> observations_new(observations.size());
        for (int j = 0; j < int(observations.size()); ++j) {
            double x_ob = observations[j].x;
            double y_ob = observations[j].y;

            // new coordinates
            double new_x = cos(theta) * x_ob - sin(theta) * y_ob + x;
            double new_y = sin(theta) * x_ob + cos(theta) * y_ob + y;
            observations_new[j] = LandmarkObs{j, new_x, new_y};
        }

        // find the association for the observations
        dataAssociation(map_in_range, observations_new);
            
        particles[i].weight = 1.0;              // reset weight to 1.0
        for (int j = 0; j < int(observations_new.size()); ++j) {
            // using mult-variate Gaussian distribution
            double dx = observations_new[j].x - map_in_range[observations_new[j].id].x;
            double dy = observations_new[j].y - map_in_range[observations_new[j].id].y;

            // update the weight
            double w = (1.0 / (2.0*M_PI*sigma_x*sigma_y)) * exp(-(dx*dx /(2.0*sigma_x*sigma_x) + dy*dy/(2.0*sigma_y*sigma_y)));
            if (w == 0) {
                particles[i].weight *= 0.00001;
            } else {
                particles[i].weight *= w;
            }
        }
        weights[i] = particles[i].weight;
    }


}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    double max_w = numeric_limits<double>::min();

    for (int i = 0; i < num_particles; ++i) {
        if (max_w < weights[i]) {
            max_w = weights[i];
        }
    }

    // Random number distributions
    uniform_real_distribution<double> dist1(0.0, max_w);
    uniform_int_distribution<int> dist2(0, num_particles-1);


    int index = dist2(gen);                                 // initial index
    double beta = 0.0;                                      // initial variable of random number
    vector<Particle> resample_particles(num_particles);     // new particles from resampling
    // resampling 
    for (int i = 0; i < num_particles; ++i) {
        beta += dist1(gen) * 2;
        while (beta > weights[index]) {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        
        resample_particles[i] = particles[index];
    }

    // assign the new particles array
    particles = resample_particles;
}



Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
