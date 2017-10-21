function [data,item_num]=loadMNISTdata(filename)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  FUNCTION                                                                                           %                            
%    - This is to function load training and test labels and images that are from MNIST dataset       %
%                                                                                                     %
%  IUPUT                                                                                              %
%    - filename                                                                                       %
%                                                                                                     %
%  OUTPUT                                                                                             %
%    - data: labels or images(in column-wise organized order)                                         %
%    - item_num; number of labels or images                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   file=fopen(filename,'rb');% read as binary
   magic_num=fread(file,1,'int32','ieee-be');% specify a big-endian ordering to display
   item_num=fread(file,1,'int32','ieee-be');
% read label file
if magic_num==2049
   data=fread(file,inf,'uchar');
   fclose(file);
end
% read image file
if magic_num==2051
   row_num=fread(file,1,'int32','ieee-be');
   column_num=fread(file,1,'int32','ieee-be');
   data=fread(file,inf,'uchar');
   fclose(file);
% change the pixel vectors from row-wise organization to column-wise organization
   data=reshape(data,row_num,column_num,item_num);
   data=permute(data,[2 1 3]);
   vector_length=row_num*column_num;
   data=reshape(data,vector_length,item_num);
   data=1-double(data)/255;% change to double to avoid overfloat  
end
end