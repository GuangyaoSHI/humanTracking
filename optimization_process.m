domeRadius = 3
[x,y,z] = sphere(40);
xEast  = domeRadius * x +5;
yNorth = domeRadius * y +5;
zUp    = domeRadius * z;
zUp(zUp < 0) = 0;
figure('Renderer','opengl')
surf(xEast, yNorth, zUp,'FaceColor','yellow','FaceAlpha',0.8)
axis equal
hold on
xEast  = domeRadius * x +15;
yNorth = domeRadius * y +5;
surf(xEast, yNorth, zUp,'FaceColor','yellow','FaceAlpha',0.8)
hold on

% [X,Y,Z] = sphere(10);
% [U,V,W] = surfnorm(X,Y,Z);
% quiver3(X,Y,Z,U,V,W,0)
% axis equal

%predicted human trajectory 
x_human = 0:1:9;
y_human = ones(1,10);
z_human = ones(1,10);
plot3(x_human, y_human, z_human)
quiver3(x_human(1:9), y_human(1:9), z_human(1:9), x_human(2:10)-x_human(1:9), ...
    y_human(2:10)-y_human(1:9), z_human(2:10)-z_human(1:9), 0)
hold on

dims = size(trajs)
for i = 1:1:dims(1)
    X = trajs(i, :, 1);
    X = X(:);
    Y = trajs(i, :, 2);
    Y = Y(:);
    Z = trajs(i, :, 3);
    Z = Z(:);
    plot3(X, Y, Z);
    text(X(end), Y(end), Z(end), string(i))
    hold on
end

