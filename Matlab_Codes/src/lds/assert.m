
function assert( assertion, message, mode )


%ASSERT Prints message and takes action if assertion is false.
% ASSERT(ASSERTION, MESSAGE, MODE) If mode is 'error', assert
% issues an error message and then exits. Note that the error
% message will come from assert, but the trace information will
% show which function called the assert. If mode is 'warning',
% assert issues a warning message and then continues. If mode
% is 'debug', assert displays a message and inserts a breakpoint in
% the calling function immediately after the assert. This will
% put the user in debug mode.
%
%=================================================================
% Note that 'debug' mode will not work correctly if it is called
% by a subfunction of a function. A subfunction is a function
% that has been defined below another function in the same file.
% A subfunction is only visible to the other functions in the same
% file. See 'help function' for details'.
%=================================================================


if nargin < 1
   error( 'assert: not enough input arguments.' );
end
if( assertion )
   % If assertion is true, do nothing.
   return;
end
if( nargin < 2 )
   % Default message.
   message = 'Assertion failure';
end
if( nargin < 3 )
   % Default mode is error.
   mode = 'error';
end


switch lower(mode)
   case 'error'
      error( message );
   case 'warning'
      warning( message );
   case 'debug'
      disp( message );
      functions = dbstack;
      dbstop( functions(2).name, num2str(1+functions(2).line) );
end




