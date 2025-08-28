classdef my_fitness
    properties
        n_dim        	
        n_obj        	
        lb          	
        ub          	
        maximize 	
    end
    methods
        function obj = my_fitness()
            obj.n_obj = 3;
            obj.n_dim = 3;
            obj.lb = [22, 100, 0.3];
            obj.ub = [26, 140, 0.5];
            obj.maximize = false;
        end

        function y = calculation(~, x)
            
            f1 = -326.86-27.29*x(1)-13.65*x(3)-25.44*x(1)*x(2);
            f2 = -269.45+23.03*x(1)+13.9*x(2)-17.98*x(3);
            f3 = -2.12+0.3245*x(1)+0.1873*x(2)+0.11*x(1)*x(2)+0.1028*x(1)*x(3)+0.3821*x(2).^2+0.1529*x(3).^2-0.5222*x(1)*x(2).^2;
        
            y = [f1, f2, f3];
        end
    end
end