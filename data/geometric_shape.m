clear all
g = figure()
max_i = 3;
n1 = 3;
n2 = 4;
index = 1;
tiledlayout(n2,n1, 'Padding', 'none', 'TileSpacing', 'compact'); 
for k=1:n2

    for i = 1:max_i
        min_x = 0; max_x = 10;
        min_y = 0; max_y = 10;
        
        x1 = rand() * (max_x - min_x) ;
        x2 = x1 + rand() * (max_x - min_x) + 2;
        y1 = rand() * (max_y - min_y)  ;
        y2 = y1 + rand() * (max_y - min_y)  + 2;

        for j = 1:n1
            ax = subplot(n2,n1,(k-1)*n1+j);
            index = index + 1;
            grid off
    
        
        m = 0; % rectangle
%         m = 1; % circle

        rectangle('Position',[x1,x2,y1,y2],'Linewidth', 2, 'Curvature',[m,m])

        set(ax, 'box', 'on', 'Visible', 'on', 'xtick', [], 'ytick', [])
       
        end
    end
    
    
end

pos = get(g, 'Position');
pos = [1000         200         800         500]
set(g, 'Position', pos)


saveas(g, "1.png")
