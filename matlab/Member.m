classdef Member
    %{
    Member - Represents an individual member/solution in the population
    
    Attributes:
        position : array
            Current position in the search space
        fitness : float
            Fitness value of the current position
    %}
    
    properties
        position
        fitness
    end
    
    methods
        function obj = Member(position, fitness)
            %{
            Member constructor - Initialize a Member with position and fitness
            
            Inputs:
                position : array
                    Position vector in search space
                fitness : float
                    Fitness value of the position
            %}
            
            obj.position = position;
            obj.fitness = fitness;
        end
        
        function new_member = copy(obj)
            %{
            copy - Create a deep copy of the Member
            
            Returns:
                new_member : Member
                    A new Member object with copied position and fitness
            %}
            
            new_member = Member(obj.position, obj.fitness);
        end
        
        function result = gt(obj, other)
            %{
            gt - Greater than comparison based on fitness values
            
            Inputs:
                other : Member
                    Another Member to compare with
                    
            Returns:
                result : bool
                    true if this member's fitness is greater than the other's
            %}
            
            if isa(other, 'Member')
                result = obj.fitness > other.fitness;
            else
                result = false;
            end
        end
        
        function result = lt(obj, other)
            %{
            lt - Less than comparison based on fitness values
            
            Inputs:
                other : Member
                    Another Member to compare with
                    
            Returns:
                result : bool
                    true if this member's fitness is less than the other's
            %}
            
            if isa(other, 'Member')
                result = obj.fitness < other.fitness;
            else
                result = false;
            end
        end
        
        function result = eq(obj, other)
            %{
            eq - Equality comparison based on fitness values
            
            Inputs:
                other : Member
                    Another Member to compare with
                    
            Returns:
                result : bool
                    true if this member's fitness equals the other's
            %}
            
            if isa(other, 'Member')
                result = obj.fitness == other.fitness;
            else
                result = false;
            end
        end
        
        function str = char(obj)
            %{
            char - String representation of the Member
            
            Returns:
                str : string
                    Formatted string showing position and fitness
            %}
            
            str = sprintf('Position: %s - Fitness: %.6f', mat2str(obj.position), obj.fitness);
        end
    end
end
