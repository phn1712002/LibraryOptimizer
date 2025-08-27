classdef BatMember < Member
    %{
    BatMember - Represents a bat individual in the Bat Algorithm
    
    Attributes:
        position : array
            Current position in the search space
        fitness : float
            Fitness value of the current position
        frequency : float
            Frequency used for velocity update
        velocity : array
            Velocity vector of the bat
        loudness : float
            Loudness parameter (controls acceptance of new solutions)
        pulse_rate : float
            Pulse emission rate (controls random walk probability)
    %}
    
    properties
        frequency
        velocity
        loudness
        pulse_rate
    end
    
    methods
        function obj = BatMember(position, fitness, frequency, velocity, loudness, pulse_rate)
            %{
            BatMember constructor - Initialize a BatMember with position, fitness, and bat-specific parameters
            
            Inputs:
                position : array
                    Position vector in search space
                fitness : float
                    Fitness value of the position
                frequency : float
                    Frequency parameter (default: 0.0)
                velocity : array
                    Velocity vector (default: zeros)
                loudness : float
                    Loudness parameter (default: 1.0)
                pulse_rate : float
                    Pulse rate parameter (default: 0.5)
            %}
            
            % Call parent constructor
            obj@Member(position, fitness);
            
            % Set default values if not provided
            if nargin < 3
                obj.frequency = 0.0;
            else
                obj.frequency = frequency;
            end
            
            if nargin < 4
                obj.velocity = zeros(size(position));
            else
                obj.velocity = velocity;
            end
            
            if nargin < 5
                obj.loudness = 1.0;
            else
                obj.loudness = loudness;
            end
            
            if nargin < 6
                obj.pulse_rate = 0.5;
            else
                obj.pulse_rate = pulse_rate;
            end
        end
        
        function new_bat = copy(obj)
            %{
            copy - Create a deep copy of the BatMember
            
            Returns:
                new_bat : BatMember
                    A new BatMember object with copied properties
            %}
            
            new_bat = BatMember(...
                obj.position, ...
                obj.fitness, ...
                obj.frequency, ...
                obj.velocity, ...
                obj.loudness, ...
                obj.pulse_rate ...
            );
        end
        
        function str = char(obj)
            %{
            char - String representation of the BatMember
            
            Returns:
                str : string
                    Formatted string showing position, fitness, and bat parameters
            %}
            
            str = sprintf('Position: %s - Fitness: %.6f - Frequency: %.3f - Loudness: %.3f - Pulse Rate: %.3f', ...
                mat2str(obj.position), obj.fitness, obj.frequency, obj.loudness, obj.pulse_rate);
        end
    end
end
