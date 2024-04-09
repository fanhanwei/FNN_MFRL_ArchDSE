''' modify skopt/optimizer/optimizer.py line 268-282 as follows '''

        # normalize space if GP regressor
        if isinstance(self.base_estimator_, GaussianProcessRegressor):
            dimensions = normalize_dimensions(dimensions)
        self.space = Space(dimensions)
        
        self._initial_samples = None
        
        if "orthogonal" == initial_point_generator:
            self._initial_point_generator = cook_initial_point_generator("sobol")
            def get_orthogonal_array():
                parameters = []
                range_list = [bound[1]-bound[0]+1 for bound in self.space.bounds]
                for index in range_list:
                    parameters.append([i for i in range(index)])
                from allpairspy import AllPairs
                samples = []
                for i, pairs in enumerate(AllPairs(parameters, n=3)):
                    print("orthogonal_array case id {:2d}: {}".format(i, pairs))
                    samples.append(pairs)
                return samples
            self._initial_samples = get_orthogonal_array()
        elif "TED" == initial_point_generator:
            self._initial_point_generator = cook_initial_point_generator("sobol")
            def random_ted(size):
                import itertools
                from random import randint
                from sklearn.gaussian_process.kernels import RBF
                design_space = [[4,5,6], [1,2,3,4], [7,8,9,10,11], [1,2,3,4], [1,2,3,4,5], 
                    [1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5], [1,2], [1,2], [1,2,3,4,5]]
                K = list(itertools.product(*design_space))
                m = size # init training set size
                Nrted = 59 # according to original paper
                u = 0.1 # according to original paper
                length_scale = 0.1 # according to original paper

                f = RBF(length_scale=length_scale)

                def F_kk(K):
                    dis_list = []
                    for k_i in K:
                        for k_j in K:
                            dis_list.append(f(np.atleast_2d(k_i), np.atleast_2d(k_j)))
                    return np.array(dis_list).reshape(len(K), len(K))

                K_tilde = []
                for i in range(m):
                    M = [K[randint(0,len(K)-1)] for _ in range(Nrted)]
                    M = M + K_tilde
                    F = F_kk(M)
                    denoms=[F[-i][-i] + u for i in range(len(K_tilde))]
                    for i in range(len(denoms)):
                        for j in range(len(M)):
                            for k in range(len(M)):
                                F[j][k] -= (F[j][i] * F[k][i]) / denoms[i]
                    assert len(M) == F.shape[0]
                    k_i = M[np.argmax([np.linalg.norm(F[i])**2 / (F[i][i] + u) for i in range(len(M))])] # find i that maximaize norm-2(column i of F)
                    K_tilde.append(k_i)
                return K_tilde
            self._initial_samples = random_ted(n_initial_points)
            self._initial_samples = [[s[0]-4, s[1]-1, s[2]-7, s[3]-1, s[4]-1, s[5]-1, s[6]-1, s[7]-1, s[8]-1, s[9]-1, s[10]-1] for s in self._initial_samples]
        else:
            self._initial_point_generator = cook_initial_point_generator(
                initial_point_generator)

            if self._initial_point_generator is not None:
                
                transformer = self.space.get_transformer()
                self._initial_samples = self._initial_point_generator.generate(
                    self.space.dimensions, n_initial_points,
                    random_state=self.rng.randint(0, np.iinfo(np.int32).max))
                self.space.set_transformer(transformer)
