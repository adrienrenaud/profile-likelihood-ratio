"""A Python tool for profile likelihood ratio.



Usage:
    Profile_likelihood_ratio:
        mu = 0.; sigma = 1.; n_data = 1000
        data = np.random.normal(mu, sigma, n_data)
        plr = Profile_likelihood_ratio(data=data, model='m2_log_likelihood_gaus', debug_level=0)
                                        plr.profile_likelihood_ratio_confidence_interval(poi_cl=[0.954499736104, 0.682689492137])
        plr.compute_common_stat_on_data()
        plr.profile_likelihood_ratio_curve()
        plr.plr_res.plot_profile_likehood_curve_and_confidence_interval()
    Profile_likelihood_ratio_Factory:
        plr_fatory = Profile_likelihood_ratio_Factory(n_exp=1000, n_data=1000, 
                                                        data_generator_name='gaus',
                                                        model='m2_log_likelihood_gaus',
                                                        confidence_level=[0.954499736104, 0.682689492137],
                                                        parameter_of_interest='mu',
                                                        debug_level=0, output_tag='bar')
        df = plr_fatory.run()
"""



import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize as sp_minimize
from scipy.optimize import newton
from scipy.stats import chi2 as sp_chi2
from scipy.stats import norm as sp_norm

import subprocess
from timeit import default_timer as timer
import sys





class Profile_likelihood_ratio(object):
    """Compute profile likelihood ratio.
    
    Initialized with a model and some data.
    Compute MLE of the parameters with unconstraint_fit().
    Compute profile likelihood ratio with profile_likelihood_ratio_curve().
    Compute confidence interval for the parameter of interest (poi) 
    using Wilks' theorem with profile_likelihood_ratio_confidence_interval().
    Results are stored in a instance of the Profile_likelihood_ratio_result class.
    """
    
    def __init__(self, data=np.array([], dtype=float), model='m2_log_likelihood_gaus', debug_level=0, tag='foo'):
        """Creates a Profile_likelihood_ratio.
        
        Args:
            data: data to fit.
            model: a string referencing a model, 
                needs to be available in set_score_function().
            debug_level: an integer to set the verbosity
            tag: a string to be added to the output file name
        """
        self.debug_level = debug_level
        self.tag = tag
        self.model = model
        self.log_sqrt_2pi = np.log(np.sqrt(2*np.pi))
        self.plr_res = Profile_likelihood_ratio_result()
        self.constraints = ()

        self.set_data(data)
        self.set_score_function(self.model)
        
                
    def __call__(self, params):
        """To make a Profile_likelihood_ratio callable.
        Wrapper around score_function()
        Args: 
            params: the parameters of the model
        Returns:
            self.score_function(params): the model evaluated at params
        """
        return self.score_function(params)
        
        
    def set_data(self, data):
        """Set data and n_data
        
        Args:
            data: the data
        """
        if self.debug_level>0:
            print '::: Data type:', type(data)
            try:
                print '::: Data element dtype:', data.dtype
            except AttributeError:
                print '::: Data element has no dtype attribute...'
                
        self.data = data
        self.n_data = len(data)
        
        
    def set_score_function(self, model='m2_log_likelihood_gaus'):
        """Set self.score_function to the desired model.
        Set number of parameters and a dict of {'parameter_name': parameter_number}.
        Need to be edited by user to reference a new model.
        
        Args:
            model: a string referencing a model. 
        """
        if model=='m2_log_likelihood_gaus':
            self.n_params = 2
            self.initial_values = [1.]*self.n_params
            self.params_dict = {'mu': 0, 'sigma': 1}
            self.score_function = self.m2_log_likelihood_gaus
        else:
            print '::: Unknow function type, exiting...'
            sys.exit(1)

            
    def reset_constraints(self):
        """Delete the constraints on the parameters."""
        self.constraints = ()
        
    def set_constraint_eq(self, param_name='0', value=1.):
        """Set an equality constraint on a parameter.
        Args:
            param_name: the name of parameter.
            value: the value of the parameter.
        """
        param_num = self.params_dict[param_name]
        self.constraints += ({'type': 'eq', 'fun': lambda x : x[param_num] - value},)
        
        
    def resest_initial_values(self):
        """Reset the parameters initial values"""
        self.initial_values = [1.]*self.n_params
        
    def set_initial_values(self, param_name='0', value=1.):
        """Set the initial value of a parameter.
        Args:
            param_name: the name of parameter.
            value: the value of the parameter.
        """
        param_num = self.params_dict[param_name]
        self.initial_values[param_num] = value
        

    def minimize(self):
        """Minimise self.score_function,
        given the initial values and the constraints
        using scipy.optimize.minimize.
        
        Returns:
            res: the scipy.optimize.OptimizeResult (a dict).
        """
        if self.debug_level>0:
            print '-'*90
            print '::: initial values:', self.initial_values
            print '::: constraints:', self.constraints
        res = sp_minimize(self, self.initial_values, constraints=self.constraints, )

        if self.debug_level>0:
            print res
        return res
    

    def unconstraint_fit(self, poi_name='mu'):
        """Unconstraint fit to get MLE.
        
        Sets:
            self.plr_res.poi_name
            self.plr_res.poi_mle
            self.plr_res.score_poi_mle
            self.plr_res.hessian_poi_mle
        Args:
            poi_name: name of the parameter of interest.
        Returns:
            res_hat: the scipy.optimize.OptimizeResult (a dict).
        
        """
        self.plr_res.poi_name = poi_name
        poi_param_num = self.params_dict[poi_name]
 
        self.reset_constraints()
        self.resest_initial_values()
        res_hat = self.minimize() 
        
        if not res_hat.success:
            if self.debug_level>0:
                print '::: Minimization Failed in unconstraint_fit!!!!!!'
                print res_hat
        self.plr_res.poi_mle = res_hat.x[poi_param_num]
        self.plr_res.score_poi_mle = res_hat.fun
        self.plr_res.hessian_poi_mle = res_hat.hess_inv[poi_param_num, poi_param_num]
        
        return res_hat
        
        
    def profile_likelihood_ratio_curve(self, poi_name='mu', poi_n_steps=50, poi_range_n_sigma=3.):
        """Compute the profile_likelihood_ratio_curve.
            
        Args:
            poi_name: name of the parameter of interest.
            poi_n_steps: number of points in the curve.
            poi_range_n_sigma: number of std around the MLE for the curve range,
                std of the poi is estimated using the Hessian (std_poi = sqrt(2*H^-1) for -2 ln(L)).
        returns:
            self.plr_res: Profile_likelihood_ratio_result instance.
        Sets:
            self.plr_res.poi_name
            self.plr_res.pois
            self.plr_res.poi_scores
            self.plr_res.plr_curve
        """
        ### get MLE
        self.plr_res.poi_name = poi_name
        res_hat = self.unconstraint_fit(poi_name)
        
        ### set poi range for PLR from Hessian 
        ### std_poi = sqrt(2*H^-1) for -2 ln(L)
        std_poi = np.sqrt(2*self.plr_res.hessian_poi_mle)
        poi_start =  self.plr_res.poi_mle - poi_range_n_sigma * std_poi 
        poi_stop = self.plr_res.poi_mle + poi_range_n_sigma * std_poi 
        poi_range = np.linspace(poi_start, poi_stop, poi_n_steps, endpoint=True)
        
        ### Estimate L around MLE
        for i, poi in enumerate(poi_range):
            self.reset_constraints()
            self.set_constraint_eq(poi_name, poi)
            res = self.minimize()
            if not res.success:
                print '::: Minimization Failed around MLE!!!!!!'
                print res
            self.plr_res.pois.append(poi)
            self.plr_res.scores_poi.append(res.fun)

        self.plr_res.compute_plr_curve()

        return self.plr_res        
        
        
    def profile_likelihood_ratio_confidence_interval(self, poi_name='mu', poi_cl=[0.954499736104, 0.682689492137]):
        """Compute the confidence intrerval using profile likelihood ratio
        and wilks theorem: -2*log(PLR) as a chi2_1 distribution.
        CI are found using scipy.optimize.newton, finding poi that verify
        -2*log(PLR(poi))= sp_chi2.ppf(cl, 1).
            
        Args:
            poi_name: name of the parameter of interest.
            poi_cl: a list of confidence levels.
        returns:
            self.plr_res: Profile_likelihood_ratio_result instance.
        Sets:
            self.plr_res.poi_name
            self.plr_res.cl
            self.plr_res.cl_delta_chi2
            self.plr_res.cl_n_sigma
            self.plr_res.ci_poi_min
            self.plr_res.ci_poi_max
        """        
        ### get MLE
        self.plr_res.poi_name = poi_name
        res_hat = self.unconstraint_fit(poi_name)
        
        ### set poi range for PLR from Hessian 
        ### std_poi = sqrt(2*H^-1) for -2 ln(L)
        ### -2*log(PLR) as a chi2_1 distribution (chi2=1 => 68%, chi2=4 => 95%)
        std_poi = np.sqrt(2*self.plr_res.hessian_poi_mle)

        ### Estimate L around MLE
        def func(poi, poi_name, delta_chi2):
            self.reset_constraints()
            self.resest_initial_values()
            self.set_constraint_eq(poi_name, poi)
            res = self.minimize()
            if not res.success:
                print '::: Minimization Failed in finding Confidence Interval!!!!!!'
                print res
            offset = self.plr_res.score_poi_mle + delta_chi2
            return self(res.x) - offset
            
        for i, cl in enumerate(poi_cl):
            
            n_sigma = sp_norm.ppf((1+cl)/2)
            delta_chi2 = sp_chi2.ppf(cl, 1)
            
            poi_ci_guess =  self.plr_res.poi_mle + n_sigma * std_poi 
            try:
                max_ci = newton(func, poi_ci_guess, args=(poi_name, delta_chi2), maxiter=1000)
            except RuntimeError:
                print '::: Retry finding Confidence Interval with tolerance 0.1'
                max_ci = newton(func, poi_ci_guess, args=(poi_name, delta_chi2), maxiter=1000, tol=0.1)
                
            poi_ci_guess =  self.plr_res.poi_mle - n_sigma * std_poi 
            try:
                min_ci = newton(func, poi_ci_guess, args=(poi_name, delta_chi2), maxiter=1000)
            except RuntimeError:
                print '::: Retry finding Confidence Interval with tolerance 0.1'
                min_ci = newton(func, poi_ci_guess, args=(poi_name, delta_chi2), maxiter=1000, tol=0.1)
            
            self.plr_res.cl.append(cl)
            self.plr_res.cl_delta_chi2.append(delta_chi2)
            self.plr_res.cl_n_sigma.append(n_sigma)
            self.plr_res.ci_poi_min.append( min_ci )
            self.plr_res.ci_poi_max.append( max_ci )
            
            if self.debug_level>0:
                print '::: Newton res:', cl, min_ci, max_ci
            
        return self.plr_res 
    

    def compute_common_stat_on_data(self):
        """Compute common statistics on the data"""
        n_data = len(self.data)
        sum_data = self.data.sum()
        mean_data = sum_data/n_data
        rs_data = (self.data - mean_data).sum()
        rss_data = ((self.data - mean_data)**2).sum()
        var_data = rss_data/n_data
        std_data = np.sqrt(var_data)
        corr_var_data = rss_data/(n_data - 1)
        corr_std_data = np.sqrt(corr_var_data)

        self.plr_res.n_data = n_data
        self.plr_res.sum_data = sum_data
        self.plr_res.mean_data = mean_data
        self.plr_res.rs_data = rs_data
        self.plr_res.rss_data = rss_data
        self.plr_res.var_data =  var_data
        self.plr_res.std_data =  std_data
        self.plr_res.corr_var_data = corr_var_data
        self.plr_res.corr_std_data = corr_std_data
        
        return self.plr_res
        
        
    def m2_log_likelihood_gaus(self, params):
        """The model.
        User can implement his own model.
        
        Args:
            params: model parameters
        Return:
            f1+f2: -2*log(Likelihood(params|data))
            
        """
        ## params = [mu, sigma]
        f1 = 2 * self.n_data * ( np.log(abs(params[1])) + self.log_sqrt_2pi)
        f2_vect = (self.data - params[0])**2 / params[1]**2
        f2 = f2_vect.sum()
        return f1 + f2

        
        
        
        
        
        
        
class Profile_likelihood_ratio_result(object):
    """A class to store the results of a Profile_likelihood_ratio instance."""
    def __init__(self):
        """
        Sets:
            poi_name: name of the parameter of interest (poi).
            poi_mle: maximum likelihood estimate (mle) of poi.
            score_poi_mle: likelihood (li) at poi mle.
            hessian_poi_mle: hessian at poi mle.
            pois: values of poi (for plr curve).
            socres_poi: values of li at self.pois (for plr curve).
            plr_curve: values of plr at self.pois (for plr curve).
            cl: list of confidence intervals (cl).
            cl_delta_chi2: list of delta_chi2 at self.cl (scipy.chi2.ppf(cl, 1)).
            cl_n_sigma: list of n_sigma at self.cl (scipy.norm.ppf((1+cl)/2)).
            cl_poi_max: list of uper bounds of sel.cl.
            cl_poi_min: list of lower bounds of sel.cl.
        """
        ## set by profile_likelihood_ratio_curve and profile_likelihood_ratio_ci
        self.poi_name = 'mu'
        self.poi_mle = 100.
        self.score_poi_mle = 100.
        self.hessian_poi_mle = 100.
        
        ## set by profile_likelihood_ratio_curve
        self.pois = []
        self.scores_poi = []
        self.plr_curve = []
        
        ## set by profile_likelihood_ratio_ci
        self.cl = []
        self.cl_delta_chi2 = []
        self.cl_n_sigma = []
        self.ci_poi_max = []
        self.ci_poi_min = []
        
        ## set by compute_common_stat_on_data
        self.n_data = 0
        self.sum_data = 0.
        self.mean_data = 0.
        self.rs_data = 0.
        self.rss_data = 0.
        
        
    def compute_plr_curve(self):
        """Compute the profile_likehood_curve."""
        self.plr_curve = [score_poi - self.score_poi_mle for score_poi in self.scores_poi]

        
    def plot_profile_likehood_curve_and_confidence_interval(self, output_path='./plr_curve'):
        """Plot the profile_likehood_curve and the confidence intervals."""
        poi_name = self.poi_name
        
        fig = plt.figure(figsize=(7,7))#plt.figure(figsize = (7,7))
        axe = fig.add_subplot(111)
        axe.plot(self.pois, self.plr_curve, color='k')
        for delta_chi2 in self.cl_delta_chi2:
            axe.axhline(delta_chi2, linewidth=1, color='k')
        axe.scatter(self.ci_poi_min, self.cl_delta_chi2, marker='x', color='black', s=40, linewidth=2)
        axe.scatter(self.ci_poi_max, self.cl_delta_chi2, marker='x', color='black', s=40, linewidth=2)
        
        axe.set_xlabel(r'$\mathregular{%s}$'%poi_name, fontsize='xx-large')
        axe.set_ylabel(r'$\mathregular{-2 \times \ln(\Lambda)}$', fontsize='xx-large')
        axe.set_ylim(bottom=0)
        
        textstr = '$\mathregular{%s}$ = %.4f'%(poi_name, self.poi_mle)
        for cl, min_, max_ in zip(self.cl, self.ci_poi_min, self.ci_poi_max):
            textstr+= '\n%i%% CL: [%.4f, %.4f]'%(cl*100, min_, max_)
        props = dict(boxstyle='square', facecolor='w', alpha=0.5)
        axe.text(1.05, 0.95, textstr, transform=axe.transAxes, fontsize=16, verticalalignment='top', bbox=props)
        
        fig.tight_layout()
        plt.show()
        fig.savefig(output_path + '.pdf')

    
    
    
    
    
    
class Profile_likelihood_ratio_Factory(object):
    """A factory of Profile_likelihood_ratio.
    
    Used to do coverage studies, 
    or distribution sampling of PLR.
    """
    def __init__(self, n_exp=10, n_data=10000, 
                       data_generator_name='gaus',
                       model='m2_log_likelihood_gaus', parameter_of_interest='mu',
                       confidence_level=[0.954499736104, 0.682689492137],
                       debug_level=0, output_tag='foo',):
        """Creates a Profile_likelihood_ratio_Factory.
        
        Args:
            n_exp: number of experiments to be done.
            n_data: number of data points.
            data_generator_name: string referencing the name of the data_generator.
            model: string referencing the model.
            parameter_of_interest: name of the parameter of interest.
            confidence_level: list of confidence levels.
            debug_level: verbosity level.
            output_tag: name of the directory where results are saved.
        """
                       
        self.confidence_level = confidence_level
        self.poi_name = parameter_of_interest
        self.model = model
        self.data_generator_name = data_generator_name
        self.debug_level = debug_level
        self.n_exp = n_exp
        self.n_data = n_data
        # self.true_mu = true_mu
        # self.true_sigma = true_sigma
        self.output_tag = output_tag
        self.output_file = 'mod_fact_out_ndata_%i_nexp_%i'%(self.n_data, self.n_exp)
        self.dump_attributes()

        self.array_plr_res = np.empty(self.n_exp, dtype=Profile_likelihood_ratio_result)
        self.dict_of_array_result = {}
        self.data_frame_result = None

        self.set_data_generator(self.data_generator_name)
        
        subprocess.call(['mkdir', 'results'])
        subprocess.call(['mkdir', 'results/'+self.output_tag])
        
        
    def dump_attributes(self):
        """Dump the attributes of the instance."""
        attributes = self.__dict__
        max_lenght_attr = max([len(attr) for attr in attributes.keys()])
        for attr,v in attributes.iteritems(): 
            st = '::: %s'%attr + ' '*(max_lenght_attr - len(attr)) + ' :'
            print st, v
        
        
    def set_data_generator(self, data_generator_name):
        """Set the data generator.
        User can edit to implement his own generator.
        
        Args:
            data_generator_name: string referencing the name of the data_generator.
        """
        if data_generator_name=='gaus':
            self.data_generator = self.gaus_generator
        else:
            print '::: Unknow data generator name, exiting...'
            sys.exit(1)
            
            
    def run(self):
        """Run the Factory.
        
        Returns:
            self.data_frame_result: a pandas dataframe containing the results.
        """
        np.random.seed(2)
        
        start = timer()
        for i in range(self.n_exp):
            if not i%100:
                print '::: Iteration %i (%i to go)'%(i, self.n_exp - i)
            self.array_plr_res[i] = self.runPLR()
        end = timer()
        print '::: Time to run %i iterations:'%self.n_exp, end - start
        
        self.create_dict_of_array_result()
        self.create_data_frame_result()
        
        return self.data_frame_result
    
    
    def runPLR(self):
        """Run a Profile_likelihood_ratio.
        
        Returns:
            res: a Profile_likelihood_ratio_result instance.
        """
        # data = np.random.normal(self.true_mu, self.true_sigma, self.n_data)
        data = self.data_generator()
        
        model = Profile_likelihood_ratio(data=data, 
                                         model=self.model,
                                         debug_level=self.debug_level
                                         )
        model.profile_likelihood_ratio_confidence_interval(poi_name=self.poi_name, 
                                                           poi_cl=self.confidence_level
                                                           )
        model.compute_common_stat_on_data()
        res = model.plr_res
        
        return res
        
        
    def create_dict_of_array_result(self):
        """Create a dict containing the results (before creating a dataframe)
        
        Returns:
            self.dict_of_array_result: a dict containing the results.
        """
        start = timer()
        plr_res_attr = self.array_plr_res[0].__dict__.keys()

        for k in plr_res_attr:
            self.dict_of_array_result[k] = []
        for k in self.true_params_dict.keys():
            self.dict_of_array_result[k] = []
                
        for i,res in enumerate(self.array_plr_res):
            for k in self.true_params_dict.keys():
                self.dict_of_array_result[k] = self.true_params_dict[k]
            # self.dict_of_array_result['true_mu'].append(self.true_mu)
            # self.dict_of_array_result['true_sigma'].append(self.true_sigma)
            for k in plr_res_attr:
                self.dict_of_array_result[k].append(res.__dict__[k])

        end = timer()
        print '::: Time to create result dict:', end - start
        return self.dict_of_array_result
        
        
    def create_data_frame_result(self):
        """Create a pandas dataframe containing the results.
        
        Returns:
            self.data_frame_result: a pandas dataframe containing the results.
        """
        start = timer()
        self.data_frame_result = pd.DataFrame(self.dict_of_array_result)
        end = timer()
        print '::: Time to create result data frame:', end - start
        
        self.data_frame_result.to_pickle('results/'+self.output_tag + '/' + self.output_file)
        return self.data_frame_result
    
    
    def gaus_generator(self):
        """"The data generator.
        User can implement is hown generator.
        
        Returns:
            data: the data to fit.
        Sets:
            self.true_params_dict: a dict containing the parameters true values.
        """
        mu = 0.
        sigma = 1.
        self.true_params_dict = {'true_mu': mu, 'true_sigma': sigma}
        data = np.random.normal(mu, sigma, self.n_data)
        return data
        
   