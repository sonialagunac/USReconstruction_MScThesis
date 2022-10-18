function drawAnnotation(app)
    n = 100;
    % when Tension=0 the class of Cardinal spline is known as Catmull-Rom spline
    Tension=0; 

    Px = single(app.lesionPoints(1,:));
    Px = [Px(end-1:end), Px, Px(1:2)]; 
    Py = single(app.lesionPoints(2,:));
    Py = [Py(end-1:end), Py, Py(1:2)];
    

    hold(app.UIAxesB,"on");
    hold(app.UIAxesSoS,"on");
    curve = [];
    for k=1:length(Px)-3

        [XiYi]=crdatnplusoneval([Px(k),Py(k)],[Px(k+1),Py(k+1)],[Px(k+2),Py(k+2)],[Px(k+3),Py(k+3)],Tension,n);

        % % XiYi is 2D interpolated data
        curve = [curve, XiYi];
        % Between each pair of control points plotting n+1 values of first two rows of XiYi 
        plot(app.UIAxesB,XiYi(1,:),XiYi(2,:),'yellow','linewidth',1) % interpolated data
        plot(app.UIAxesB,Px,Py,'rx','linewidth',1)          % control points
        plot(app.UIAxesSoS,XiYi(1,:),XiYi(2,:),'black','linewidth',1) % interpolated data
        plot(app.UIAxesSoS,Px,Py,'wx','linewidth',1)          % control points
    end
    
    bimage = NaN * ones(size(app.UIAxesSoS.Children(end,1).CData));
    curve(1,:) = round((curve(1,:)  - app.UIAxesSoS.Children(end,1).XData(1)) * size(bimage,2) / (app.UIAxesSoS.Children(end,1).XData(end) - app.UIAxesSoS.Children(end,1).XData(1)),0);
    curve(2,:) = round(curve(2,:) * size(bimage,1) / (app.UIAxesSoS.Children(end,1).YData(end) - app.UIAxesSoS.Children(end,1).YData(1)),0);
    Px = round((Px  - app.UIAxesSoS.Children(end,1).XData(1)) * size(bimage,2) / (app.UIAxesSoS.Children(end,1).XData(end) - app.UIAxesSoS.Children(end,1).XData(1)),0);
    Py = round(Py * size(bimage,1) / (app.UIAxesSoS.Children(end,1).YData(end) - app.UIAxesSoS.Children(end,1).YData(1)),0);
    
    xmin = round(min(curve(1,:)),0);
    xmax = floor(max(curve(1,:)));
    areacoordinates = NaN* ones((xmax-xmin)+1,3);
    for ll = xmin : xmax
        cidx = find((curve(1,:) >= ll) & (curve(1,:) < ll+1));
        areacoordinates(ll-xmin+1,:) = [ll, min(curve(2,cidx)), max(curve(2,cidx))];
    end
    for jj = 1:length(areacoordinates)
        bimage(areacoordinates(jj,2):areacoordinates(jj,3),areacoordinates(jj,1)) = 1;
    end
    bimage = bimage .* app.UIAxesSoS.Children(end,1).CData;

    bimage1 = NaN * ones(size(app.UIAxesSoS.Children(end,1).CData));
    xmin = round(min(curve(2,:)),0);
    xmax = floor(max(curve(2,:)));
    areacoordinates = NaN* ones((xmax-xmin)+1,3);
    for ll = xmin : xmax
        cidx = find((curve(2,:) >= ll) & (curve(2,:) < ll+1));
        if isempty(cidx) && (ll > 1)
            cidx = find((curve(2,:) >= ll-1) & (curve(2,:) < ll));
        end
        areacoordinates(ll-xmin+1,:) = [ll, min(curve(1,cidx)), max(curve(1,cidx))];
    end
    for jj = 1:length(areacoordinates)
        bimage1(areacoordinates(jj,1),areacoordinates(jj,2):areacoordinates(jj,3)) = 1;
    end
     maskedImage = bimage1 .* bimage;
        
            figure            
            imagesc(maskedImage)
            hold on
            plot(curve(1,:),curve(2,:),'b','linewidth',2) % interpolated data
            plot(Px,Py,'ro','linewidth',1)          % control points

            figure            
            imagesc(bimage1)
            hold on
            plot(curve(1,:),curve(2,:),'b','linewidth',2) % interpolated data
            plot(Px,Py,'ro','linewidth',1)          % control points
end
